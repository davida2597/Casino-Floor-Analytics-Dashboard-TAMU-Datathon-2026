# src/scorecard.py
import numpy as np
import pandas as pd
from .io_utils import con
from .config import START, END

# src/scorecard.py — replace SQL and TREND_SQL
SQL = """
WITH d AS (
    SELECT dn.*, cal.scheduled_seconds, cal.scheduled_hours,
           cal.is_holiday, cal.day_of_week
    FROM dn JOIN cal USING (gaming_date)
    WHERE dn.gaming_date BETWEEN DATE '{start}' AND DATE '{end}'
      AND dn.active
      AND dn.location_id IS NOT NULL   -- 1,018 pre-install/post-retire rows [2]
)
SELECT
    machine_id,
    COUNT(*)                                          AS days_on_floor,
    MIN(gaming_date)                                  AS first_day,
    MAX(gaming_date)                                  AS last_day,
    SUM(scheduled_hours)                              AS scheduled_hours,
    SUM(available_hours)                              AS available_hours,
    SUM(available_hours) / NULLIF(SUM(scheduled_hours), 0) AS availability,
    SUM(coin_in_usd)                                  AS coin_in,
    SUM(payout_usd)                                   AS payout,
    SUM(house_win_usd)                                AS house_win,
    SUM(theo_usd)                                     AS theo,
    SUM(house_win_usd) - SUM(theo_usd)                AS luck_residual,
    SUM(house_win_usd) / NULLIF(SUM(coin_in_usd), 0)  AS observed_hold,
    SUM(theo_usd)      / NULLIF(SUM(coin_in_usd), 0)  AS expected_hold,
    SUM(coin_in_usd)   / NULLIF(SUM(available_hours), 0) AS coin_in_per_avail_hr,
    SUM(theo_usd)      / NULLIF(SUM(available_hours), 0) AS theo_per_avail_hr,
    SUM(theo_usd)      / NULLIF(SUM(scheduled_hours), 0) AS theo_per_sched_hr,
    SUM(carded_coin_in_usd) / NULLIF(SUM(coin_in_usd), 0) AS carded_share,
    SUM(faults_opened)                                AS faults_opened_view,
    COUNT(DISTINCT location_id)                       AS n_locations,
    COUNT(DISTINCT game_setting_id)                   AS n_game_settings
FROM d
GROUP BY machine_id
"""

TREND_SQL = """
WITH d AS (
    SELECT machine_id, gaming_date, coin_in_usd, theo_usd,
           available_hours, cal.day_of_week
    FROM dn JOIN cal USING (gaming_date)
    WHERE active AND location_id IS NOT NULL
      AND gaming_date BETWEEN DATE '{start}' AND DATE '{end}'
)
SELECT
    machine_id,
    SUM(CASE WHEN gaming_date >= DATE '{end}' - INTERVAL 28 DAY
             THEN coin_in_usd END)
      / NULLIF(SUM(CASE WHEN gaming_date >= DATE '{end}' - INTERVAL 28 DAY
                        THEN available_hours END), 0)       AS cipah_last28,
    SUM(CASE WHEN gaming_date <  DATE '{end}' - INTERVAL 28 DAY
             THEN coin_in_usd END)
      / NULLIF(SUM(CASE WHEN gaming_date <  DATE '{end}' - INTERVAL 28 DAY
                        THEN available_hours END), 0)       AS cipah_prior,
    SUM(CASE WHEN day_of_week IN (5, 6) THEN coin_in_usd END)
      / NULLIF(SUM(CASE WHEN day_of_week IN (5, 6)
                        THEN available_hours END), 0)       AS cipah_weekend,
    SUM(CASE WHEN day_of_week NOT IN (5, 6) THEN coin_in_usd END)
      / NULLIF(SUM(CASE WHEN day_of_week NOT IN (5, 6)
                        THEN available_hours END), 0)       AS cipah_weekday
FROM d GROUP BY machine_id
"""


def _pct(s: pd.Series) -> pd.Series:
    """Percentile rank 0-100, NaN-safe."""
    return s.rank(pct=True, na_option="keep") * 100


def build_scorecard(start=START, end=END, c=None) -> pd.DataFrame:
    c = c or con()
    base = c.execute(SQL.format(start=start, end=end)).df()
    trend = c.execute(TREND_SQL.format(start=start, end=end)).df()

    meta = c.execute("""
        SELECT machine_id, manufacturer, theme AS game_title, install_date, retire_date
        FROM machines
    """).df()

    # Dominant game setting and location, weighted by available hours — not a
    # permanent attribute, because machines move [2].
    dom = c.execute(f"""
        WITH d AS (
            SELECT machine_id, game_setting_id, location_id, zone_id, bank_id,
                   available_hours
            FROM dn
            WHERE active AND location_id IS NOT NULL
              AND gaming_date BETWEEN DATE '{start}' AND DATE '{end}'
        ),
        g AS (
            SELECT machine_id, game_setting_id,
                   SUM(available_hours) AS h,
                   ROW_NUMBER() OVER (PARTITION BY machine_id
                                      ORDER BY SUM(available_hours) DESC) AS rk
            FROM d GROUP BY 1, 2
        ),
        z AS (
            SELECT machine_id, zone_id, bank_id,
                   SUM(available_hours) AS h,
                   SUM(available_hours) / SUM(SUM(available_hours)) OVER (
                       PARTITION BY machine_id) AS share,
                   ROW_NUMBER() OVER (PARTITION BY machine_id
                                      ORDER BY SUM(available_hours) DESC) AS rk
            FROM d GROUP BY 1, 2, 3
        )
        SELECT g.machine_id, g.game_setting_id AS dom_game_setting,
               z.zone_id AS dom_zone, z.bank_id AS dom_bank,
               z.share  AS dom_zone_share
        FROM g JOIN z USING (machine_id)
        WHERE g.rk = 1 AND z.rk = 1
    """).df()

    math_tbl = c.execute("SELECT * FROM gm").df()

    sc = (base
          .merge(trend, on="machine_id", how="left")
          .merge(meta, on="machine_id", how="left")
          .merge(dom, on="machine_id", how="left")
          .merge(math_tbl, left_on="dom_game_setting",
                 right_on="game_setting_id", how="left", suffixes=("", "_gm")))

    sc["downtime_hours"] = (sc["scheduled_hours"] - sc["available_hours"]).clip(lower=0)
    sc["trend_pct"] = (sc["cipah_last28"] / sc["cipah_prior"] - 1) * 100
    sc["weekend_lift"] = sc["cipah_weekend"] / sc["cipah_prior"].where(
        sc["cipah_weekday"].isna(), sc["cipah_weekday"])
    sc["hold_gap_pp"] = (sc["observed_hold"] - sc["expected_hold"]) * 100

    # Exposure flag: too little play to conclude anything.
    sc["low_exposure"] = (
        (sc["days_on_floor"] < 45) |
        (sc["coin_in"] < sc["coin_in"].quantile(0.05)) |
        (sc["available_hours"] < 500)
    )

    for col in ["theo_per_avail_hr", "coin_in_per_avail_hr", "theo_per_sched_hr",
                "availability", "house_win", "theo"]:
        sc[f"pct_{col}"] = _pct(sc[col])

    return sc


def add_peer_percentiles(sc: pd.DataFrame, keys=("dom_game_setting",),
                         cols=("theo_per_avail_hr", "coin_in_per_avail_hr",
                               "availability")) -> pd.DataFrame:
    """Percentile within a peer group, not just floor-wide."""
    sc = sc.copy()
    suffix = "_".join(k.replace("dom_", "") for k in keys)
    g = sc.groupby(list(keys), dropna=False)
    for col in cols:
        sc[f"peer_{suffix}_pct_{col}"] = g[col].transform(_pct)
        sc[f"peer_{suffix}_median_{col}"] = g[col].transform("median")
    return sc


def peer_distribution(sc: pd.DataFrame, machine_id: str, col: str,
                      mode: str = "floor") -> pd.Series:
    """Return the comparison distribution for a dot plot."""
    row = sc.loc[sc.machine_id == machine_id].iloc[0]
    if mode == "floor":
        return sc[col].dropna()
    key = {"game_setting": "dom_game_setting", "volatility": "volatility_index",
           "zone": "dom_zone", "manufacturer": "manufacturer"}[mode]
    return sc.loc[sc[key] == row[key], col].dropna()