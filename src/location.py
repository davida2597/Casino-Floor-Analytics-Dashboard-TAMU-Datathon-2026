# src/location.py  (complete file)
import numpy as np
import pandas as pd
from .io_utils import con
from .config import START, END


def location_rates(start=START, end=END, c=None) -> pd.DataFrame:
    """Raw earning rate per seat, plus which machines occupied it."""
    c = c or con()
    return c.execute(f"""
        SELECT location_id,
               ANY_VALUE(zone_id)  AS zone_id,
               ANY_VALUE(bank_id)  AS bank_id,
               COUNT(DISTINCT machine_id)                      AS n_machines,
               SUM(available_hours)                            AS avail_hours,
               SUM(coin_in_usd)                                AS coin_in,
               SUM(theo_usd)                                   AS theo,
               SUM(coin_in_usd) / NULLIF(SUM(available_hours),0) AS cipah,
               SUM(theo_usd)    / NULLIF(SUM(available_hours),0) AS tpah
        FROM dn
        WHERE active AND location_id IS NOT NULL
          AND gaming_date BETWEEN DATE '{start}' AND DATE '{end}'
        GROUP BY location_id
        ORDER BY cipah DESC
    """).df()


def two_way_effects(start=START, end=END, c=None, min_hours=50) -> dict:
    """
    Additive decomposition of log earning rate into machine, seat and
    day effects, fit by alternating least squares (no statsmodels needed).

        log(cipah) ≈ mu + machine_effect + location_effect + dow_effect

    Weighted by available hours so a 2-hour day cannot outvote a 24-hour day.
    """
    c = c or con()
    d = c.execute(f"""
        SELECT machine_id, location_id, zone_id, bank_id,
               gaming_date, cal.day_of_week,
               available_hours, coin_in_usd
        FROM dn JOIN cal USING (gaming_date)
        WHERE active AND location_id IS NOT NULL
          AND available_hours > 0.5
          AND gaming_date BETWEEN DATE '{start}' AND DATE '{end}'
    """).df()

    d["rate"] = d["coin_in_usd"] / d["available_hours"]
    d = d[d["rate"] > 0].copy()
    d["y"] = np.log(d["rate"])
    d["w"] = d["available_hours"]

    mu = np.average(d["y"], weights=d["w"])
    eff = {k: pd.Series(0.0, index=d[k].unique())
           for k in ["machine_id", "location_id", "day_of_week"]}

    for _ in range(40):
        for k in ["machine_id", "location_id", "day_of_week"]:
            others = [o for o in eff if o != k]
            resid = d["y"] - mu - sum(d[o].map(eff[o]) for o in others)
            num = (resid * d["w"]).groupby(d[k]).sum()
            den = d.groupby(d[k])["w"].sum()
            new = num / den
            eff[k] = new - np.average(new, weights=den.reindex(new.index))

    fitted = mu + sum(d[k].map(eff[k]) for k in eff)
    resid = d["y"] - fitted
    r2 = 1 - np.average(resid ** 2, weights=d["w"]) / \
             np.average((d["y"] - mu) ** 2, weights=d["w"])

    loc_hours = d.groupby("location_id")["available_hours"].sum()
    mach_hours = d.groupby("machine_id")["available_hours"].sum()

    loc_df = (pd.DataFrame({"location_id": eff["location_id"].index,
                            "loc_effect": eff["location_id"].values})
              .assign(avail_hours=lambda x: x.location_id.map(loc_hours),
                      loc_multiplier=lambda x: np.exp(x.loc_effect))
              .query("avail_hours >= @min_hours")
              .sort_values("loc_effect", ascending=False))

    mach_df = (pd.DataFrame({"machine_id": eff["machine_id"].index,
                             "machine_effect": eff["machine_id"].values})
               .assign(avail_hours=lambda x: x.machine_id.map(mach_hours),
                       machine_multiplier=lambda x: np.exp(x.machine_effect)))

    zone_of = d.drop_duplicates("location_id").set_index("location_id")
    loc_df["zone_id"] = loc_df["location_id"].map(zone_of["zone_id"])
    loc_df["bank_id"] = loc_df["location_id"].map(zone_of["bank_id"])

    return {"locations": loc_df, "machines": mach_df,
            "dow": eff["day_of_week"].sort_index(), "mu": mu, "weighted_r2": r2,
            "n_movers": int((d.groupby("machine_id")["location_id"]
                             .nunique() > 1).sum())}


def movers_evidence(start=START, end=END, c=None) -> pd.DataFrame:
    """
    Before/after for machines that actually moved — the cleanest natural
    experiment in the dataset, and the honest basis for a "move it" estimate.
    """
    c = c or con()
    return c.execute(f"""
        WITH spans AS (
            SELECT machine_id, location_id, zone_id,
                   MIN(gaming_date) AS span_start,
                   MAX(gaming_date) AS span_end,
                   SUM(available_hours) AS avail_hours,
                   SUM(coin_in_usd) / NULLIF(SUM(available_hours),0) AS cipah,
                   SUM(theo_usd)    / NULLIF(SUM(available_hours),0) AS tpah
            FROM dn
            WHERE active AND location_id IS NOT NULL
              AND gaming_date BETWEEN DATE '{start}' AND DATE '{end}'
            GROUP BY machine_id, location_id, zone_id
            HAVING SUM(available_hours) >= 100
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY machine_id
                                         ORDER BY span_start) AS seq,
                   COUNT(*) OVER (PARTITION BY machine_id) AS n_spans
            FROM spans
        )
        SELECT a.machine_id,
               a.location_id AS from_loc, b.location_id AS to_loc,
               a.zone_id AS from_zone,   b.zone_id AS to_zone,
               a.cipah AS cipah_before,  b.cipah AS cipah_after,
               b.cipah / NULLIF(a.cipah, 0) - 1 AS cipah_change,
               a.avail_hours AS hours_before, b.avail_hours AS hours_after,
               b.span_start AS move_date
        FROM ranked a JOIN ranked b
          ON a.machine_id = b.machine_id AND b.seq = a.seq + 1
        WHERE a.n_spans > 1
        ORDER BY cipah_change DESC
    """).df()