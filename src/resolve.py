# src/resolve.py
"""Normalize release-0.9.0 column names into canonical views.

Creates three views every other module reads from:
  cal - gaming_date, scheduled_seconds, scheduled_hours, day_of_week,
        is_weekend, is_holiday
  gm  - game_setting_id, rtp, volatility_index, avg_bet_usd, denom_usd
  dn  - machine_id, gaming_date, location_id, zone_id, bank_id,
        game_setting_id, coin_in_usd, payout_usd, house_win_usd, theo_usd,
        available_hours, active, carded_coin_in_usd, faults_opened,
        events_active, lag7_coin_in, lag28_coin_in

Release 0.9.0-draft promises stable tables and columns but warns values may
change before 1.0.0, so this is the single file to edit if names shift [2].
"""


def cols(c, view) -> list[str]:
    return [r[0] for r in c.execute(f"DESCRIBE {view}").fetchall()]


def pick(available, *candidates, label="", required=False, default=None):
    """Exact match first, then unique-substring match, else default."""
    low = {x.lower(): x for x in available}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for cand in candidates:
        hits = sorted({orig for lc, orig in low.items() if cand.lower() in lc})
        if len(hits) == 1:
            return hits[0]
    if required:
        raise KeyError(
            f"{label or candidates[0]!r}: none of {candidates} found.\n"
            f"Available: {sorted(available)}")
    return default


def expr(col, cast=None, fallback="NULL"):
    if not col:
        return fallback
    return f"CAST({col} AS {cast})" if cast else col


# ------------------------------------------------------------------ calendar
def build_cal(c) -> dict:
    cc = cols(c, "calendar_day")
    sec = pick(cc, "scheduled_seconds", "gaming_day_seconds", "day_seconds",
               "length_seconds", "seconds")
    sec_expr = f"CAST({sec} AS DOUBLE)" if sec else "86400.0"

    # Derive weekday from the date: ISODOW is 1=Mon..7=Sun.
    dow_expr = "CAST(ISODOW(gaming_date) - 1 AS INTEGER)"
    wknd = pick(cc, "is_weekend")
    hol = pick(cc, "is_holiday", "holiday_flag")
    hol_name = pick(cc, "holiday_name", "holiday", "holiday_label")

    if hol:
        hol_expr = f"CAST({hol} AS BOOLEAN)"
    elif hol_name:
        hol_expr = f"({hol_name} IS NOT NULL AND CAST({hol_name} AS VARCHAR) <> '')"
    else:
        hol_expr = "FALSE"

    c.execute(f"""
        CREATE OR REPLACE VIEW cal AS
        SELECT gaming_date,
               {sec_expr}                                   AS scheduled_seconds,
               {sec_expr} / 3600.0                           AS scheduled_hours,
               {dow_expr}                                    AS day_of_week,
               {f'CAST({wknd} AS BOOLEAN)' if wknd else 'ISODOW(gaming_date) >= 6'}
                                                            AS is_weekend,
               {hol_expr}                                    AS is_holiday
        FROM calendar_day
    """)
    return {"scheduled_seconds": sec or "defaulted to 86400",
            "is_holiday": hol or hol_name or "FALSE (absent)"}


# ----------------------------------------------------------------- game_math
# ----------------------------------------------------------------- game_math
def _works(c, expr_sql: str, table="game_math") -> bool:
    """Validate an expression against the table before baking it into a view."""
    try:
        c.execute(f"SELECT {expr_sql} FROM {table} LIMIT 1").fetchall()
        return True
    except Exception:
        return False


def build_gm(c) -> dict:
    gc = cols(c, "game_math")
    types = {r[0]: str(r[1]).upper()
             for r in c.execute("DESCRIBE game_math").fetchall()}

    key  = pick(gc, "config_id", "game_setting_id", "setting_id", "paytable_id",
                label="game_math key", required=True)
    rtp  = pick(gc, "configured_rtp", "rtp", "rtp_pct", "published_rtp")
    var  = pick(gc, "payout_multiplier_variance", "payout_variance", "variance")
    volc = pick(gc, "volatility_class", "volatility", "volatility_level")
    minb = pick(gc, "min_bet_cents", "min_bet_usd", "min_bet")
    maxb = pick(gc, "max_bet_cents", "max_bet_usd", "max_bet")
    opts = pick(gc, "bet_options_cents", "bet_options")
    hit  = pick(gc, "hit_probability", "hit_rate", "hit_frequency")
    mult = pick(gc, "max_multiplier", "top_multiplier")
    theme = pick(gc, "theme", "game_theme")

    # RTP may ship as 0.92 or 92 [2]
    rtp_expr = (f"CASE WHEN CAST({rtp} AS DOUBLE) > 1.5 "
                f"THEN CAST({rtp} AS DOUBLE)/100.0 ELSE CAST({rtp} AS DOUBLE) END"
                ) if rtp else "NULL"

    # --- average bet: prefer the mean of the actual bet options ---
    # _cents columns are integer cents, _usd are dollars [2]
    def to_usd(col):
        return f"CAST({col} AS DOUBLE)/100.0" if col and col.endswith("_cents") \
               else f"CAST({col} AS DOUBLE)"

    bet_expr, bet_src = "NULL", "none"
    if opts:
        cands = [
            (f"list_aggregate({opts}, 'avg')/100.0", "list_aggregate(avg)"),
            (f"list_avg({opts})/100.0", "list_avg"),
            (f"list_aggregate(list_transform(str_split(CAST({opts} AS VARCHAR), ','), "
             f"x -> CAST(trim(x) AS DOUBLE)), 'avg')/100.0", "parsed string list"),
        ]
        for e, src in cands:
            if _works(c, e):
                bet_expr, bet_src = e, f"{opts} via {src}"
                break
    if bet_expr == "NULL" and minb and maxb:
        bet_expr = f"({to_usd(minb)} + {to_usd(maxb)})/2.0"
        bet_src = f"midpoint of {minb}/{maxb} (assumption)"
    elif bet_expr == "NULL" and minb:
        bet_expr, bet_src = to_usd(minb), f"{minb} only (assumption)"

    # --- volatility: a numeric variance if published, else an ordinal label ---
    if var:
        vol_expr = f"CAST({var} AS DOUBLE)"          # variance of payout multiplier
        sd_expr  = f"SQRT(CAST({var} AS DOUBLE))"
        vol_src  = var
    elif volc:
        sd_expr = (f"CASE lower(CAST({volc} AS VARCHAR)) "
                   f"WHEN 'low' THEN 1.5 WHEN 'medium' THEN 3.0 WHEN 'med' THEN 3.0 "
                   f"WHEN 'high' THEN 6.0 WHEN 'very_high' THEN 9.0 "
                   f"WHEN 'very high' THEN 9.0 ELSE 3.0 END")
        vol_expr = f"POWER({sd_expr}, 2)"
        vol_src = f"{volc} (ordinal -> assumed SD)"
    else:
        vol_expr, sd_expr, vol_src = "9.0", "3.0", "defaulted SD=3"

    # src/resolve.py  (build_gm, completed)
    c.execute(f"""
        CREATE OR REPLACE VIEW gm AS
        SELECT {key}                               AS game_setting_id,
               {rtp_expr}                          AS rtp,
               1 - ({rtp_expr})                    AS expected_hold_cfg,
               {vol_expr}                          AS payout_var,
               {sd_expr}                           AS payout_sd,
               {sd_expr}                           AS volatility_index,
               {f'CAST({volc} AS VARCHAR)' if volc else "'unknown'"}
                                                   AS volatility_class,
               {bet_expr}                          AS avg_bet_usd,
               {to_usd(minb) if minb else 'NULL'}  AS min_bet_usd,
               {to_usd(maxb) if maxb else 'NULL'}  AS max_bet_usd,
               {expr(hit,  'DOUBLE')}              AS hit_probability,
               {expr(mult, 'DOUBLE')}              AS max_multiplier,
               {f'CAST({theme} AS VARCHAR)' if theme else "'unknown'"} AS theme
        FROM game_math
    """)

    return {"key": key, "rtp": rtp, "payout_var": vol_src,
            "avg_bet": bet_src, "volatility_class": volc or "absent",
            "hit_probability": hit or "absent", "theme": theme or "absent"}

# --------------------------------------------------------------------- daily
# src/resolve.py  (continued)

# --------------------------------------------------------------------- daily
# src/resolve.py
# --------------------------------------------------------------------- daily
def build_dn(c) -> dict:
    """Canonical daily view. Chain is dn_base -> dn_theo -> dn, because a
    DuckDB view cannot reference itself."""
    if not c.execute(
        "SELECT COUNT(*) FROM duckdb_views() WHERE view_name = 'daily'"
    ).fetchone()[0]:
        raise RuntimeError(
            "view 'daily' not registered — check TABLES in io_utils.py points "
            "at data/1_prepared_views/prepared_daily_view.parquet")

    dc = cols(c, "daily")
    m = {}
    m["machine_id"]  = pick(dc, "machine_id", required=True, label="machine_id")
    m["gaming_date"] = pick(dc, "gaming_date", required=True, label="gaming_date")
    m["location_id"] = pick(dc, "location_id", "slot_id", "position_id")
    m["zone_id"]     = pick(dc, "zone_id", "zone")
    m["bank_id"]     = pick(dc, "bank_id", "bank")
    m["game_setting_id"] = pick(dc, "config_id", "game_config_id",
                                "game_setting_id", "setting_id", "paytable_id")
    m["coin_in_usd"] = pick(dc, "coin_in_usd", "total_coin_in_usd", "coin_in",
                            required=True, label="coin_in_usd")
    m["payout_usd"]  = pick(dc, "payout_usd", "total_payout_usd", "payout")
    m["house_win_usd"] = pick(dc, "house_win_usd", "total_house_win_usd",
                              "net_win_usd", "house_win")
    m["theo_usd"]    = pick(dc, "theo_usd", "theoretical_win_usd", "theo_win_usd")
    m["available_hours"] = pick(dc, "available_hours", "avail_hours",
                                "uptime_hours", required=True,
                                label="available_hours")
    m["active"]      = pick(dc, "active", "is_active")
    m["carded_coin_in_usd"] = pick(dc, "carded_coin_in_usd")
    m["faults_opened"] = pick(dc, "faults_opened", "fault_count", "n_faults")
    m["events_active"] = pick(dc, "events_active", "event_active", "has_event")
    m["lag7_coin_in"]  = pick(dc, "lag7_coin_in_usd", "coin_in_usd_lag7",
                              "lag_7_coin_in_usd", "lag7_coin_in")
    m["lag28_coin_in"] = pick(dc, "lag28_coin_in_usd", "coin_in_usd_lag28",
                              "lag_28_coin_in_usd", "lag28_coin_in")

    # House win = coin-in - payout if not published [2]
    hw = (m["house_win_usd"] or
          (f"({m['coin_in_usd']} - {m['payout_usd']})"
           if m["payout_usd"] else "NULL"))

    c.execute(f"""
        CREATE OR REPLACE VIEW dn_base AS
        SELECT {m['machine_id']}                        AS machine_id,
               {m['gaming_date']}                       AS gaming_date,
               {expr(m['location_id'], fallback='NULL')}      AS location_id,
               {expr(m['zone_id'], fallback='NULL')}          AS zone_id,
               {expr(m['bank_id'], fallback='NULL')}          AS bank_id,
               {expr(m['game_setting_id'], fallback='NULL')}  AS game_setting_id,
               CAST({m['coin_in_usd']} AS DOUBLE)       AS coin_in_usd,
               {expr(m['payout_usd'], 'DOUBLE')}        AS payout_usd,
               CAST({hw} AS DOUBLE)                     AS house_win_usd,
               {expr(m['theo_usd'], 'DOUBLE')}          AS theo_usd,
               CAST({m['available_hours']} AS DOUBLE)   AS available_hours,
               {f'CAST({m["active"]} AS BOOLEAN)' if m['active'] else 'TRUE'}
                                                        AS active,
               {expr(m['carded_coin_in_usd'], 'DOUBLE')} AS carded_coin_in_usd,
               {expr(m['faults_opened'], 'DOUBLE')}      AS faults_opened,
               {expr(m['events_active'], 'DOUBLE')}      AS events_active,
               {expr(m['lag7_coin_in'], 'DOUBLE')}       AS lag7_coin_in,
               {expr(m['lag28_coin_in'], 'DOUBLE')}      AS lag28_coin_in
        FROM daily
    """)

    # theo = coin-in x (1 - RTP) when the view doesn't publish it [2]
    if m["theo_usd"] is None and m["game_setting_id"]:
        c.execute("""
            CREATE OR REPLACE VIEW dn_theo AS
            SELECT b.* REPLACE (b.coin_in_usd * (1 - gm.rtp) AS theo_usd)
            FROM dn_base b LEFT JOIN gm USING (game_setting_id)
        """)
        m["theo_usd"] = "derived from gm.rtp"
    else:
        c.execute("CREATE OR REPLACE VIEW dn_theo AS SELECT * FROM dn_base")

    # Peer grouping needs some game identity; fall back to game_title
    if m["game_setting_id"] is None:
        c.execute("""
            CREATE OR REPLACE VIEW dn AS
            SELECT t.* REPLACE (mm.theme AS game_setting_id)
            FROM dn_theo t LEFT JOIN machines mm USING (machine_id)
        """)
        m["game_setting_id"] = "FALLBACK: machines.theme"
    else:
        c.execute("CREATE OR REPLACE VIEW dn AS SELECT * FROM dn_theo")

    return m

# ------------------------------------------------------------------ driver
def build_all(c, verbose=True) -> dict:
    """Build cal, gm, dn in dependency order (dn reads gm for derived theo)."""
    resolved = {"cal": build_cal(c), "gm": build_gm(c)}
    resolved["dn"] = build_dn(c)
    if verbose:
        for view, mapping in resolved.items():
            print(f"[resolve] {view}: {mapping}")
    return resolved
