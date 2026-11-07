# src/downtime.py  (continued / complete file)
import numpy as np
import pandas as pd
from .io_utils import con
from .config import START, END, SUBSTITUTION_RETENTION


def _gaming_date(ts: pd.Series) -> pd.Series:
    """Gaming day runs 6 a.m. to 6 a.m., so shift back six hours [2]."""
    return (pd.to_datetime(ts, errors="coerce") - pd.Timedelta(hours=6)).dt.date


def fault_metrics(start=START, end=END, c=None) -> pd.DataFrame:
    c = c or con()
    ops = c.execute("SELECT * FROM operations").df()

    open_col  = next(x for x in ops.columns if x.startswith("opened_at_local")
                     or x == "opened_at_utc")
    ops["gaming_date"] = _gaming_date(ops[open_col])
    ops = ops[(ops.gaming_date >= start) & (ops.gaming_date <= end)].copy()

    # Duration: the table ships opened_at_utc and available_at_utc (when the
    # machine came back). Use UTC for both ends so there is no timezone skew.
    dur_col = next((x for x in ["duration_hours", "downtime_hours", "repair_hours"]
                    if x in ops.columns), None)
    if dur_col is None and {"opened_at_utc", "available_at_utc"} <= set(ops.columns):
        ops["duration_hours"] = (
            pd.to_datetime(ops["available_at_utc"]) - pd.to_datetime(ops["opened_at_utc"])
        ).dt.total_seconds() / 3600
        dur_col = "duration_hours"
    if dur_col is None:
        raise KeyError(f"no duration source in operations columns: {list(ops.columns)}")

    tcol = next(x for x in ["ticket_type", "event_type", "type", "category"]
                if x in ops.columns)

    is_fault = ops[tcol].str.contains("fault|repair|jam|error", case=False, na=False)
    is_move  = ops[tcol].str.contains("move|placement", case=False, na=False)

    agg = (ops[is_fault].groupby("machine_id")
           .agg(n_faults=("machine_id", "size"),
                fault_hours=(dur_col, "sum"),
                mttr_hours=(dur_col, "mean"),
                worst_fault_hours=(dur_col, "max"))
           .reset_index())

    agg["n_moves"] = agg["machine_id"].map(
        ops[is_move].groupby("machine_id").size()).fillna(0).astype(int)

    # Repeat-offender signal: faults clustered in time suggest a real defect
    # rather than bad luck with one long repair.
    f = ops[is_fault].sort_values([ "machine_id", open_col])
    f["gap_days"] = (f.groupby("machine_id")[open_col]
                      .diff().dt.total_seconds() / 86400)
    agg["repeat_faults_14d"] = agg["machine_id"].map(
        f[f.gap_days <= 14].groupby("machine_id").size()).fillna(0).astype(int)

    agg["ticket_table"] = tcol          # provenance for the evidence appendix
    return agg


def lost_play(sc: pd.DataFrame, retention=SUBSTITUTION_RETENTION) -> pd.DataFrame:
    """
    Value of downtime. Two things matter and most teams miss the second:
      1. The machine's own earning rate while it WAS available.
      2. Substitution - a player whose machine is dead often just moves one
         seat over, so the floor loses less than the machine does.
    """
    out = sc.copy()
    out["lost_coin_in_gross"] = out["downtime_hours"] * out["coin_in_per_avail_hr"]
    out["lost_theo_gross"]    = out["downtime_hours"] * out["theo_per_avail_hr"]
    out["lost_theo_net"]      = out["lost_theo_gross"] * retention
    out["retention_assumed"]  = retention
    return out


def downtime_calendar(start=START, end=END, c=None) -> pd.DataFrame:
    """Q3: how much play was lost to downtime, and WHEN."""
    c = c or con()
    return c.execute(f"""
        WITH d AS (
            SELECT gaming_date, machine_id, available_hours,
                   coin_in_usd, theo_usd, zone_id,
                   cal.scheduled_hours AS sched_h,
                   cal.day_of_week, cal.is_holiday
            FROM dn JOIN cal USING (gaming_date)
            WHERE active AND location_id IS NOT NULL
              AND gaming_date BETWEEN DATE '{start}' AND DATE '{end}'
        ),
        rate AS (   -- each machine's own earning rate over the whole window
            SELECT machine_id,
                   SUM(coin_in_usd) / NULLIF(SUM(available_hours), 0) AS cipah,
                   SUM(theo_usd)    / NULLIF(SUM(available_hours), 0) AS tpah
            FROM d GROUP BY 1
        )
        SELECT d.gaming_date, d.day_of_week, d.is_holiday, d.zone_id,
               SUM(d.sched_h - d.available_hours)              AS downtime_hours,
               SUM((d.sched_h - d.available_hours) * r.cipah)  AS lost_coin_in,
               SUM((d.sched_h - d.available_hours) * r.tpah)   AS lost_theo,
               SUM(d.coin_in_usd)                              AS actual_coin_in,
               COUNT(DISTINCT d.machine_id)                    AS machines
        FROM d JOIN rate r USING (machine_id)
        GROUP BY 1, 2, 3, 4
        ORDER BY 1
    """).df()