# src/catalog_fallback.py
import pandas as pd

# Prices and limits exactly as published in the guide [2]. The catalogue
# gives prices and limits only — never effects — so gains stay modelled.
FALLBACK = pd.DataFrame([
    ("move_machine",       "Move one machine",                   180_000,     0,  24),
    ("move_bank",          "Move a whole bank",                  950_000,     0,   3),
    ("replace_lease",      "Replace with a leased machine",      250_000, 5_500,  20),
    ("replace_purchase",   "Replace with a purchased machine", 2_200_000,     0,  10),
    ("technician",         "Add a technician to a shift",      1_152_000,     0,   3),
    ("preventive_maint",   "Preventive maintenance, one mfr",    900_000,     0,   3),
    ("inspect",            "Inspect a machine, verify payout",   120_000,     0,  15),
    ("zone_signage",       "New signs for a zone",               750_000,     0,   2),
    ("comp_pilot",         "Comp policy pilot with control",     400_000,     0,   1),
    ("comp_rate_floorwide","Change the comp rate floor-wide",    150_000,     0,   1),
    ("promo_weekend",      "Promotion weekend",                1_500_000,     0,   2),
    ("guest_survey",       "Bigger guest survey",                350_000,     0,   1),
    ("keep_budget",        "Keep the budget",                          0,     0, 999),
], columns=["action_id", "action_name", "unit_cost_cents",
            "recurring_cost_cents_per_day", "quantity_limit"])
FALLBACK["lead_days"] = 0          # overwrite from the real catalogue