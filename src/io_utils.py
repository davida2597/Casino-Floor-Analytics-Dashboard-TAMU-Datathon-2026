# src/io_utils.py
from pathlib import Path
import duckdb
import pandas as pd

DATA = Path("data")

TABLES = {
    "daily":      DATA / "1_prepared_views/prepared_daily_view.parquet",
    "property":   DATA / "1_prepared_views/property_day.parquet",
    "player":     DATA / "1_prepared_views/player_period.parquet",
    "machines":   DATA / "2_machine_decision/machines.parquet",
    "game_math":  DATA / "2_machine_decision/game_math.parquet",
    "placements": DATA / "2_machine_decision/placements.parquet",
    "locations":  DATA / "2_machine_decision/floor_locations.parquet",
    "operations": DATA / "2_machine_decision/operations.parquet",
    "calendar_day": DATA / "2_machine_decision/calendar_day.parquet",
    "actions":    DATA / "2_machine_decision/action_catalog.parquet",
    "cost_rules": DATA / "2_machine_decision/cost_rules.parquet",
    "events":     DATA / "4_context/events.parquet",
    "traffic":    DATA / "4_context/zone_traffic_hourly.parquet",
}


# src/io_utils.py  (replace con() and add helpers)

def _cols(c, view) -> list[str]:
    return [r[0] for r in c.execute(f"DESCRIBE {view}").fetchall()]


def _pick(cols, *candidates, default=None):
    """First candidate that exists, else a literal default expression."""
    lower = {col.lower(): col for col in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return default


def _build_cal(c) -> dict:
    """
    Canonical calendar view `cal` with guaranteed columns:
        gaming_date, scheduled_seconds, scheduled_hours,
        day_of_week (0=Mon .. 6=Sun), is_weekend, is_holiday
    Derived from whatever calendar_day actually ships, so a rename in
    release 1.0.0 only breaks this one function [2].
    """
    cols = _cols(c, "calendar_day")

    sec = _pick(cols, "scheduled_seconds", "gaming_day_seconds",
                "day_seconds", "length_seconds", "seconds_in_day")
    sec_expr = f"CAST({sec} AS DOUBLE)" if sec else "86400.0"

    # Derive weekday from the date itself: always correct, never renamed.
    # isodow gives 1=Mon..7=Sun, so subtract 1 for 0=Mon..6=Sun.
    dow_expr = "CAST(ISODOW(gaming_date) - 1 AS INTEGER)"

    wknd = _pick(cols, "is_weekend")
    wknd_expr = f"CAST({wknd} AS BOOLEAN)" if wknd else \
                "ISODOW(gaming_date) >= 6"

    hol = _pick(cols, "is_holiday", "holiday_flag")
    hol_name = _pick(cols, "holiday_name", "holiday", "holiday_label")
    if hol:
        hol_expr = f"CAST({hol} AS BOOLEAN)"
    elif hol_name:
        hol_expr = f"({hol_name} IS NOT NULL AND {hol_name} <> '')"
    else:
        hol_expr = "FALSE"

    c.execute(f"""
        CREATE OR REPLACE VIEW cal AS
        SELECT gaming_date,
               {sec_expr}              AS scheduled_seconds,
               {sec_expr} / 3600.0     AS scheduled_hours,
               {dow_expr}              AS day_of_week,
               {wknd_expr}             AS is_weekend,
               {hol_expr}              AS is_holiday
        FROM calendar_day
    """)
    return {"scheduled_seconds": sec or "LITERAL 86400",
            "is_weekend": wknd or "derived from date",
            "is_holiday": hol or hol_name or "FALSE (no holiday column)"}


# src/io_utils.py
from .resolve import build_all

RESOLVED: dict = {}   # canonical-view column maps (DuckDB connections reject new attributes)

def con(verbose=True):
    c = duckdb.connect()
    for name, path in TABLES.items():
        if path.exists():
            c.execute(f"CREATE OR REPLACE VIEW {name} AS "
                      f"SELECT * FROM '{path.as_posix()}'")
    RESOLVED.update(build_all(c, verbose=verbose))
    return c


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(TABLES[name])


def schema_report() -> pd.DataFrame:
    rows = []
    for name, path in TABLES.items():
        if not path.exists():
            rows.append({"table": name, "status": "MISSING", "n_cols": 0, "columns": ""})
            continue
        df = pd.read_parquet(path)
        rows.append({
            "table": name,
            "status": "ok",
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "columns": ", ".join(df.columns),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", None)
    print(schema_report().to_string())