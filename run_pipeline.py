# run_pipeline.py
from pathlib import Path
import json
import pandas as pd

from src.io_utils import con, schema_report
from src.config import START, END, BUDGET_USD, SUBSTITUTION_RETENTION
from src import (scorecard, downtime, payout, location, diagnose,
                 demand_model, floor_forecast, actions)

OUT = Path("out"); OUT.mkdir(exist_ok=True)


def main(retention=SUBSTITUTION_RETENTION):
    c = con()
    schema_report().to_csv(OUT / "schema.csv", index=False)

    # ---- 1. scorecard + peer percentiles ----
    print("1/9  scorecard")
    sc = scorecard.build_scorecard(START, END, c)
    sc = scorecard.add_peer_percentiles(sc, keys=("dom_game_setting",))
    sc = scorecard.add_peer_percentiles(sc, keys=("dom_zone",))

    # ---- 2. faults and downtime value ----
    print("2/9  downtime")
    faults = downtime.fault_metrics(START, END, c)
    sc = sc.merge(faults[["machine_id", "n_faults", "fault_hours",
                          "mttr_hours", "repeat_faults_14d"]],
                  on="machine_id", how="left")
    sc[["n_faults", "fault_hours", "repeat_faults_14d"]] = \
        sc[["n_faults", "fault_hours", "repeat_faults_14d"]].fillna(0)
    sc = downtime.lost_play(sc, retention=retention)
    dt_cal = downtime.downtime_calendar(START, END, c)
    dt_cal.to_parquet(OUT / "downtime_calendar.parquet")

    # ---- 3. luck test with BH correction ----
    print("3/9  payout / luck")
    sc = payout.luck_test(sc)

    # ---- 4. seat vs machine effects ----
    print("4/9  location")
    loc_rates = location.location_rates(START, END, c)
    eff = location.two_way_effects(START, END, c)
    movers = location.movers_evidence(START, END, c)
    loc_rates.to_parquet(OUT / "location_rates.parquet")
    eff["locations"].to_parquet(OUT / "location_effects.parquet")
    movers.to_parquet(OUT / "movers.parquet")
    print(f"      weighted R2={eff['weighted_r2']:.3f}, "
          f"{eff['n_movers']} machines occupied >1 seat")

    # ---- 5. demand model, time-based holdout ----
    print("5/9  demand model (train Jan-Apr, test May-Jun)")
    panel = demand_model.build_panel(c)
    val = demand_model.fit_and_validate(panel)
    val["scores"].to_csv(OUT / "model_scores.csv", index=False)
    val["holdout"].to_parquet(OUT / "model_holdout.parquet")
    print(val["scores"].to_string(index=False))

    resid = demand_model.adjusted_demand(panel, val["model"])
    sc = sc.merge(resid[["machine_id", "adj_demand_resid", "adj_demand_pct",
                         "pred_cipah", "actual_cipah"]],
                  on="machine_id", how="left")

    # ---- 6. floor forecast (the required simple model) ----
    print("6/9  floor forecast")
    fp = floor_forecast.floor_panel(c)
    ff = floor_forecast.validate_floor(fp)
    ff["scores"].to_csv(OUT / "floor_forecast_scores.csv", index=False)
    ff["preds"].to_parquet(OUT / "floor_forecast_preds.parquet")
    print(ff["scores"].to_string(index=False))

    # ---- 7. diagnosis ----
    print("7/9  diagnosis")
    diag = diagnose.diagnose(sc, eff["locations"], eff["machines"],
                             faults, resid)
    diag.to_parquet(OUT / "machine_scorecard.parquet")
    audit = diagnose.bottom10_audit(diag)
    audit.to_csv(OUT / "bottom10_audit.csv", index=False)
    print(audit[["machine_id", "primary_cause", "verdict"]].to_string(index=False))
    print(diag.primary_cause.value_counts().to_string())

    # ---- 8. priced candidate actions ----
    print("8/9  candidate actions")
    cat = actions.catalog(c)
    cand = actions.build_candidates(diag, cat, eff["locations"], movers,
                                    faults, retention=retention)
    cand.to_parquet(OUT / "candidates.parquet")

    # ---- 9. portfolio under <=3 action types ----
    print("9/9  portfolio")
    best = actions.optimise(cand, BUDGET_USD, max_types=3, scenario="base")
    sens = actions.sensitivity(cand, BUDGET_USD)
    best["picks"].to_csv(OUT / "portfolio.csv", index=False)
    sens.to_csv(OUT / "portfolio_sensitivity.csv", index=False)

    summary = {
        "window": [str(START), str(END)],
        "retention_assumption": retention,
        "action_types": best["types"],
        "n_units": len(best["picks"]),
        "spend": round(best["spend"], 2),
        "unspent": round(best.get("unspent", 0), 2),
        "expected_gain": round(best["gain"], 2),
        "net": round(best["net"], 2),
        "beats_doing_nothing": bool(best["beats_doing_nothing"]),
        "model_mae": val["scores"].set_index("name")["mae_log"].to_dict(),
        "leakage_notes": val["leakage_notes"],
        "location_model_r2": round(eff["weighted_r2"], 4),
        "n_movers": eff["n_movers"],
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    write_claims_appendix(diag, audit, val, ff, eff, best, sens, retention)
    return summary


def write_claims_appendix(diag, audit, val, ff, eff, best, sens, retention):
    """Claim -> query -> result -> alternative explanation -> limitation [2]."""
    rows = []
    for _, r in audit.iterrows():
        rows.append({
            "claim": f"{r.machine_id} should not be replaced on raw house win alone"
                     if r.primary_cause != "weak_game"
                     else f"{r.machine_id} is a defensible replacement candidate",
            "query": "src/scorecard.py build_scorecard + src/payout.py luck_test "
                     "+ src/location.py two_way_effects",
            "result": (f"house win ${r.house_win:,.0f}, theo ${r.theo:,.0f}, "
                       f"luck residual ${r.luck_residual:,.0f}, availability "
                       f"{r.availability:.1%}, theo/avail-hr ${r.theo_per_avail_hr:.2f} "
                       f"(fair rank {r.fair_rank_theo_per_hr} of ~310)"),
            "alternative": r.alt_explanation,
            "limitation": "Six months of data; luck SE uses a volatility "
                          "approximation; location effects are adjusted "
                          "associations, identified off few movers",
        })

    rows.append({
        "claim": "The demand model is validated without leakage",
        "query": "src/demand_model.py fit_and_validate",
        "result": " | ".join(f"{n}: MAE(log)={m:.4f}" for n, m in
                             zip(val['scores']['name'], val['scores']['mae_log'])),
        "alternative": "A game-setting x weekday mean could match the model; it is "
                       "reported as a baseline rather than hidden",
        "limitation": "Split is Jan-Apr vs May-Jun, so seasonality beyond June "
                      "is untested; lags use shift(1) only [2]",
    })
    rows.append({
        "claim": "Downtime cost the floor real money, but less than gross lost theo",
        "query": "src/downtime.py lost_play + downtime_calendar",
        "result": f"${diag['lost_theo_gross'].sum():,.0f} gross vs "
                  f"${diag['lost_theo_net'].sum():,.0f} net at "
                  f"{retention:.0%} non-substitution",
        "alternative": "Players substitute to adjacent machines, so machine-level "
                       "loss overstates floor-level loss",
        "limitation": f"Retention of {retention:.0%} is an assumption; see "
                      "sensitivity at 20% / 45% / 80%",
    })
    rows.append({
        "claim": f"The best portfolio funds {', '.join(best['types'])}",
        "query": "src/actions.py build_candidates + optimise",
        "result": f"spend ${best['spend']:,.0f}, expected gain "
                  f"${best['gain']:,.0f}, net ${best['net']:,.0f}, "
                  f"unspent ${best.get('unspent', 0):,.0f}",
        "alternative": "Keeping the entire budget is a valid answer and is the "
                       "benchmark this portfolio is measured against [2]",
        "limitation": "Effect sizes are modelled, not experimental; the low "
                      "scenario shows whether the plan survives pessimism",
    })
    pd.DataFrame(rows).to_csv(OUT / "claims_evidence_appendix.csv", index=False)


if __name__ == "__main__":
    main()