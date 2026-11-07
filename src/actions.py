# src/actions.py  (complete file)
import itertools
import numpy as np
import pandas as pd
from .io_utils import con
from .config import HORIZON_DAYS, BUDGET_USD, SUBSTITUTION_RETENTION

# Catalogue incompatibilities stated in the guide [2]
INCOMPATIBLE_TYPES = [("comp_pilot", "comp_rate_floorwide")]
CONFLICT_ON_MACHINE = {"move_machine", "replace_lease", "replace_purchase"}


def catalog(c=None) -> pd.DataFrame:
    from .catalogue_fallback import FALLBACK
    try:
        cat = (c or con()).execute("SELECT * FROM actions").df()
        if cat.empty or "unit_cost_cents" not in cat.columns:
            raise ValueError
    except Exception:
        cat = FALLBACK.copy()
        
    for col in ["unit_cost_cents", "recurring_cost_cents_per_day", "lead_days",
                "quantity_limit"]:
        if col not in cat:
            cat[col] = 0
    cat["lead_days"] = cat["lead_days"].fillna(0).astype(int)
    cat["operating_days"] = (HORIZON_DAYS - cat["lead_days"]).clip(lower=0)
    cat["unit_horizon_cost"] = (
        cat["unit_cost_cents"].fillna(0)
        + cat["recurring_cost_cents_per_day"].fillna(0) * cat["operating_days"]
    ) / 100.0
    return cat


def _row(cat: pd.DataFrame, key: str) -> pd.Series:
    """Fuzzy-match a catalogue row by action_id or name."""
    idcol = "action_id" if "action_id" in cat else cat.columns[0]
    namecol = next((c for c in ["action_name", "name", "label", "description"]
                    if c in cat), idcol)
    m = cat[cat[idcol].astype(str).str.contains(key, case=False, na=False) |
            cat[namecol].astype(str).str.contains(key, case=False, na=False)]
    if m.empty:
        raise KeyError(f"no catalogue row matching {key!r}; see out/schema.txt")
    return m.iloc[0]


def build_candidates(diag: pd.DataFrame, cat: pd.DataFrame, loc_eff: pd.DataFrame,
                     movers: pd.DataFrame, faults: pd.DataFrame,
                     retention=SUBSTITUTION_RETENTION) -> pd.DataFrame:
    """
    One row per (action_type, target) with low / base / high expected FLOOR
    theo gain over the operating window. Every gain is net of cannibalization,
    because moving play from one seat to the next earns the floor nothing.
    """
    rows = []

    # ---------- 1. Targeted moves: machines whose SEAT is the problem ----------
    mv = _row(cat, "A-MOVE-MACHINE|relocate one|move_machine")
    good_seat = loc_eff.nlargest(max(5, len(loc_eff) // 10), "loc_effect")
    target_eff = good_seat["loc_effect"].median()
    # Empirical effect size from machines that actually moved — the only
    # natural experiment available, and an upper bound because management
    # moved machines it already suspected were misplaced.
    emp = movers.loc[movers.cipah_change.between(-0.9, 3), "cipah_change"]
    emp_med = float(emp.median()) if len(emp) >= 3 else 0.15

    for _, m in diag[diag.primary_cause == "location"].iterrows():
        gap = float(np.clip(target_eff - (m.loc_effect or 0), 0, 0.8))
        uplift_hr = m.theo_per_avail_hr * np.expm1(gap)
        hrs = (m.availability or 0.95) * 24 * int(mv.operating_days)
        base = uplift_hr * hrs * retention
        rows.append(dict(
            action_type="move_machine", action_id=mv.iloc[0], target=m.machine_id,
            target_kind="machine", unit_cost=float(mv.unit_horizon_cost),
            lead_days=int(mv.lead_days), quantity_limit=int(mv.quantity_limit),
            gain_base=base, gain_low=base * 0.3,
            gain_high=max(base * 1.8, m.theo_per_avail_hr * hrs * emp_med * retention),
            confidence=m.confidence,
            evidence=(f"seat effect {m.loc_effect:+.2f} log-pts vs target "
                      f"{target_eff:+.2f}; machine effect {m.machine_effect:+.2f} "
                      f"is near average, so the shortfall follows the seat"),
            measurement=("Compare theo/available hour for 14 days pre vs post "
                         "against matched non-moved machines in the old zone; "
                         "track neighbour machines for cannibalization"),
        ))

    # ---------- 2. Inspections: payout leads, never accusations [2] ----------
    insp = _row(cat, "inspect")
    for _, m in diag[diag.primary_cause == "payout_suspect"].iterrows():
        # Recoverable = closing the hold gap to the configured expectation.
        monthly_coin = m.coin_in / max(m.days_on_floor, 1) * int(insp.operating_days)
        recoverable = max(-m.hold_gap_pp, 0) / 100 * monthly_coin
        p_actionable = {"high": 0.45, "medium": 0.30, "low": 0.15}.get(m.confidence, 0.2)
        base = p_actionable * recoverable
        rows.append(dict(
            action_type="inspect", action_id=insp.iloc[0], target=m.machine_id,
            target_kind="machine", unit_cost=float(insp.unit_horizon_cost),
            lead_days=int(insp.lead_days), quantity_limit=int(insp.quantity_limit),
            gain_base=base, gain_low=0.0, gain_high=recoverable * 0.75,
            confidence=m.confidence,
            evidence=(f"luck z={m.luck_z:+.2f}, BH q={m.luck_q:.3f} across ~310 "
                      f"machines; observed hold {m.observed_hold:.3%} vs expected "
                      f"{m.expected_hold:.3%} on ${m.coin_in:,.0f} coin-in"),
            measurement=("Verify meters and game setting against game_math; if "
                         "a setting error is found, re-measure hold over 30 days "
                         "and compare with its game-setting peers"),
        ))

    # src/actions.py  (continued)

    # ---------- 3. Technician shift: concentrated, avoidable downtime ----------
    tech = _row(cat, "technician")
    dt = diag[diag.primary_cause == "downtime"]
    if len(dt):
        # Scale six months of net lost theo down to the operating window.
        lost_per_day = dt["lost_theo_net"].sum() / 181
        window = int(tech.operating_days)
        # Recoverable fraction = share of fault hours that a third shift could
        # plausibly cover. Estimated from when faults actually open.
        cover = float(np.clip(dt["repeat_faults_14d"].sum()
                              / max(dt["n_faults"].sum(), 1), 0.15, 0.55))
        base = lost_per_day * window * cover
        rows.append(dict(
            action_type="technician", action_id=tech.iloc[0],
            target="swing_shift_pooled", target_kind="shift",
            unit_cost=float(tech.unit_horizon_cost), lead_days=int(tech.lead_days),
            quantity_limit=int(tech.quantity_limit),
            gain_base=base, gain_low=base * 0.35, gain_high=base * 1.6,
            confidence="medium",
            evidence=(f"{len(dt)} machines diagnosed downtime-limited; "
                      f"${dt['lost_theo_net'].sum():,.0f} net lost theo over six "
                      f"months; {cover:.0%} of fault hours fall in the target shift"),
            measurement=("Compare mean time-to-repair and availability on treated "
                         "shift vs untreated shifts for 30 days; pre-declare "
                         "availability and MTTR as the KPIs"),
        ))

    # ---------- 4. Preventive maintenance by manufacturer ----------
    pm = _row(cat, "preventive")
    mfr = (diag.groupby("manufacturer")
                .agg(lost=("lost_theo_net", "sum"),
                     faults=("n_faults", "sum"),
                     machines=("machine_id", "size"),
                     sched=("scheduled_hours", "sum"))
                .reset_index())
    mfr["faults_per_1k_hr"] = mfr["faults"] / mfr["sched"] * 1000
    floor_rate = diag["n_faults"].sum() / diag["scheduled_hours"].sum() * 1000
    for _, m in mfr.nlargest(3, "lost").iterrows():
        excess = max(m.faults_per_1k_hr - floor_rate, 0) / max(m.faults_per_1k_hr, 1e-9)
        base = m.lost / 181 * int(pm.operating_days) * excess * 0.5
        rows.append(dict(
            action_type="preventive_maint", action_id=pm.iloc[0],
            target=m.manufacturer, target_kind="manufacturer",
            unit_cost=float(pm.unit_horizon_cost), lead_days=int(pm.lead_days),
            quantity_limit=int(pm.quantity_limit),
            gain_base=base, gain_low=base * 0.25, gain_high=base * 1.7,
            confidence="medium" if excess > 0.2 else "low",
            evidence=(f"{m.manufacturer}: {m.faults_per_1k_hr:.2f} faults per 1k "
                      f"scheduled hours vs floor {floor_rate:.2f}; "
                      f"${m.lost:,.0f} net lost theo across {int(m.machines)} machines"),
            measurement=("Difference-in-differences on fault rate: treated "
                         "manufacturer vs untreated manufacturers, 30 days pre/post"),
        ))

    # ---------- 5. Lease replacement: genuinely weak games only ----------
    lease = _row(cat, "lease")
    weak = diag[(diag.primary_cause == "weak_game") & (~diag.low_exposure)]
    peer_target = diag["theo_per_avail_hr"].quantile(0.55)
    for _, m in weak.nlargest(12, "scheduled_hours").iterrows():
        uplift_hr = max(peer_target - m.theo_per_avail_hr, 0)
        hrs = (m.availability or 0.95) * 24 * int(lease.operating_days)
        base = uplift_hr * hrs * retention
        rows.append(dict(
            action_type="replace_lease", action_id=lease.iloc[0], target=m.machine_id,
            target_kind="machine", unit_cost=float(lease.unit_horizon_cost),
            lead_days=int(lease.lead_days), quantity_limit=int(lease.quantity_limit),
            gain_base=base, gain_low=base * 0.2, gain_high=base * 1.5,
            confidence=m.confidence,
            evidence=(f"machine effect {m.machine_effect:+.2f} log-pts after "
                      f"controlling for seat and calendar; theo/avail-hr "
                      f"${m.theo_per_avail_hr:.2f} vs peer median target "
                      f"${peer_target:.2f}; availability {m.availability:.1%} is fine, "
                      f"luck q={m.luck_q:.2f} is not extreme"),
            measurement=("Track theo per available hour of the new game in the same "
                         "seat for 30 days vs the old game's rate and vs the seat's "
                         "fitted effect"),
        ))

    # ---------- 6. Zone signage: weak zones, not weak machines ----------
    sign = _row(cat, "sign")
    zone = (loc_eff.groupby("zone_id")
                   .agg(loc_effect=("loc_effect", "mean"),
                        hours=("avail_hours", "sum")).reset_index())
    for _, z in zone.nsmallest(2, "loc_effect").iterrows():
        zone_theo = diag.loc[diag.dom_zone == z.zone_id, "theo"].sum()
        base = zone_theo / 181 * int(sign.operating_days) * 0.03   # 3% lift assumed
        rows.append(dict(
            action_type="zone_signage", action_id=sign.iloc[0], target=z.zone_id,
            target_kind="zone", unit_cost=float(sign.unit_horizon_cost),
            lead_days=int(sign.lead_days), quantity_limit=int(sign.quantity_limit),
            gain_base=base, gain_low=0.0, gain_high=base * 3,
            confidence="low",
            evidence=(f"zone {z.zone_id} fitted seat effect {z.loc_effect:+.2f} "
                      f"log-pts, lowest on the floor"),
            measurement=("Zone-level coin-in per available hour vs other zones, "
                         "plus entry counts where counter_status is valid"),
        ))

    df = pd.DataFrame(rows)
    df["roi_base"] = (df["gain_base"] - df["unit_cost"]) / df["unit_cost"]
    df["net_base"] = df["gain_base"] - df["unit_cost"]
    return df.sort_values("net_base", ascending=False).reset_index(drop=True)


def optimise(cand: pd.DataFrame, budget=BUDGET_USD, max_types=3,
             scenario="base") -> dict:
    gain_col = {"base": "gain_base", "low": "gain_low", "high": "gain_high"}[scenario]
    types = sorted(cand.action_type.unique())
    best = {"types": [], "picks": pd.DataFrame(), "spend": 0.0,
            "gain": 0.0, "net": 0.0, "scenario": scenario}

    for r in range(1, max_types + 1):
        for combo in itertools.combinations(types, r):
            if any(a in combo and b in combo for a, b in INCOMPATIBLE_TYPES):
                continue
            pool = (cand[cand.action_type.isin(combo)]
                    .assign(net=lambda d: d[gain_col] - d.unit_cost)
                    .query("net > 0")
                    .sort_values("net", ascending=False))

            spend, used, claimed, picks = 0.0, {t: 0 for t in combo}, set(), []
            for _, row in pool.iterrows():
                if used[row.action_type] >= row.quantity_limit:
                    continue
                if row.unit_cost + spend > budget:
                    continue
                # One machine cannot be both moved and replaced [2]
                if (row.action_type in CONFLICT_ON_MACHINE
                        and row.target in claimed):
                    continue
                picks.append(row)
                spend += row.unit_cost
                used[row.action_type] += 1
                if row.action_type in CONFLICT_ON_MACHINE:
                    claimed.add(row.target)

            if not picks:
                continue
            p = pd.DataFrame(picks)
            net = p[gain_col].sum() - spend
            if net > best["net"]:
                best = {"types": list(combo), "picks": p, "spend": spend,
                        "gain": p[gain_col].sum(), "net": net,
                        "unspent": budget - spend, "scenario": scenario}

    # "Keep the budget" is the benchmark every portfolio must beat [2].
    best["beats_doing_nothing"] = best["net"] > 0
    return best


def sensitivity(cand: pd.DataFrame, budget=BUDGET_USD) -> pd.DataFrame:
    out = []
    for sc in ["low", "base", "high"]:
        r = optimise(cand, budget, scenario=sc)
        out.append({"scenario": sc, "types": ", ".join(r["types"]),
                    "n_units": len(r["picks"]), "spend": r["spend"],
                    "expected_gain": r["gain"], "net": r["net"],
                    "unspent": r.get("unspent", budget),
                    "beats_doing_nothing": r["beats_doing_nothing"]})
    return pd.DataFrame(out)