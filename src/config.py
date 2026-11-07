# src/config.py
from datetime import date

# Analysis window
START, END = date(2025, 1, 1), date(2025, 6, 30)

# Modeling split: train on earlier months, test on later ones (no random split) [2]
TRAIN_END   = date(2025, 4, 30)
HOLDOUT_BEG = date(2025, 5, 1)

# Action horizon
HORIZON_START, HORIZON_DAYS, BUDGET_USD = date(2025, 7, 1), 30, 100_000

# Fraction of lost play that does NOT migrate to a neighbouring machine.
# Downtime only costs the floor money if players leave rather than move next door.
SUBSTITUTION_RETENTION = 0.45          # sensitivity: 0.2 / 0.45 / 0.8

# Column map — update after reading out/schema.txt
COLS = dict(
    machine="machine_id",
    date="gaming_date",
    coin_in="coin_in_usd",
    payout="payout_usd",
    house_win="house_win_usd",
    theo="theo_usd",
    avail_hours="available_hours",
    active="active",
    location="location_id",
    zone="zone_id",
    bank="bank_id",
    game_setting="game_setting_id",
    rtp="rtp",
    volatility="volatility_index",
    avg_bet="avg_bet_usd",
    sched_sec="scheduled_seconds",
)