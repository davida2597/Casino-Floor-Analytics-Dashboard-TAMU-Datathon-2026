# src/demand_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from .io_utils import con
from .config import START, END, TRAIN_END, HOLDOUT_BEG

TARGET = "y_log_cipah"

FEATURES = ["game_setting_id", "rtp", "volatility_index", "avg_bet_usd",
            "zone_id", "bank_id", "day_of_week", "month", "is_holiday",
            "has_event", "machine_age_days", "lag7_cipah", "lag28_cipah"]

CATS = ["game_setting_id", "zone_id", "bank_id"]

EVENT_SQL_FALLBACK = """
    EXISTS (SELECT 1 FROM events e
            WHERE gaming_date BETWEEN
                  CAST(e.{s} AS DATE) AND CAST(e.{e_} AS DATE)) AS has_event
"""

def build_panel(c=None) -> pd.DataFrame:
    c = c or con()
    d = c.execute(f"""
        SELECT machine_id, gaming_date, zone_id, bank_id,
               location_id, game_setting_id,
               coin_in_usd, theo_usd, available_hours,
               cal.day_of_week, cal.is_holiday,
               MONTH(gaming_date) AS month,
               gm.rtp, gm.volatility_index, gm.avg_bet_usd,
               DATE_DIFF('day', m.install_date, gaming_date) AS machine_age_days,
               EXISTS (SELECT 1 FROM events e
                       WHERE gaming_date BETWEEN e.start_date AND e.end_date)
                    AS has_event
        FROM dn
        JOIN cal USING (gaming_date)
        LEFT JOIN gm USING (game_setting_id)
        LEFT JOIN machines  m  USING (machine_id)
        WHERE active AND location_id IS NOT NULL
          AND available_hours > 0.5
          AND gaming_date BETWEEN DATE '{START}' AND DATE '{END}'
    """).df()

    d["cipah"] = d["coin_in_usd"] / d["available_hours"]
    d[TARGET] = np.log1p(d["cipah"])

    # Lags are computed from PRIOR days only. For a multi-day forecast these
    # must not be refreshed with holdout-day results [2].
    d = d.sort_values(["machine_id", "gaming_date"])
    g = d.groupby("machine_id")["cipah"]
    d["lag7_cipah"]  = g.transform(lambda s: s.shift(1).rolling(7,  min_periods=3).mean())
    d["lag28_cipah"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())
    for col in CATS:
        d[col] = d[col].astype("category")
    return d


# src/demand_model.py  (continued)
def fit_and_validate(panel: pd.DataFrame) -> dict:
    tr = panel[pd.to_datetime(panel.gaming_date) <= pd.Timestamp(TRAIN_END)].dropna(subset=[TARGET])
    te = panel[pd.to_datetime(panel.gaming_date) >= pd.Timestamp(HOLDOUT_BEG)].dropna(subset=[TARGET])

    X_tr, y_tr = tr[FEATURES], tr[TARGET]
    X_te, y_te = te[FEATURES], te[TARGET]

    model = HistGradientBoostingRegressor(
        loss="absolute_error", max_depth=6, learning_rate=0.06,
        max_iter=500, early_stopping=True, validation_fraction=0.15,
        categorical_features=[FEATURES.index(c) for c in CATS],
        random_state=0,
    ).fit(X_tr, y_tr)

    # --- baselines, frozen on training data only [2] ---
    # B0: one constant (training mean rate)
    b0 = np.full(len(te), tr[TARGET].mean())
    # B1: mean by weekday
    dow_mean = tr.groupby("day_of_week")[TARGET].mean()
    b1 = te["day_of_week"].map(dow_mean).fillna(tr[TARGET].mean()).values
    # B2: mean by game_setting x weekday — the honest baseline to beat
    gs_mean = tr.groupby(["game_setting_id", "day_of_week"],
                         observed=True)[TARGET].mean()
    b2 = (te.set_index(["game_setting_id", "day_of_week"]).index.map(gs_mean)
            .to_numpy(dtype=float))
    b2 = np.where(np.isnan(b2), tr[TARGET].mean(), b2)

    pred = model.predict(X_te)

    def dollars(logv):          # back to $/available hour
        return np.expm1(logv)

    scores = pd.DataFrame([
        {"name": "model_gbm",          "mae_log": mean_absolute_error(y_te, pred),
         "mae_usd_per_hr": mean_absolute_error(dollars(y_te), dollars(pred))},
        {"name": "baseline_constant",  "mae_log": mean_absolute_error(y_te, b0),
         "mae_usd_per_hr": mean_absolute_error(dollars(y_te), dollars(b0))},
        {"name": "baseline_weekday",   "mae_log": mean_absolute_error(y_te, b1),
         "mae_usd_per_hr": mean_absolute_error(dollars(y_te), dollars(b1))},
        {"name": "baseline_gs_weekday","mae_log": mean_absolute_error(y_te, b2),
         "mae_usd_per_hr": mean_absolute_error(dollars(y_te), dollars(b2))},
    ])
    scores["beats_best_baseline"] = (
        scores["mae_log"] <= scores.loc[1:, "mae_log"].min())

    holdout = te[["machine_id", "gaming_date", "available_hours",
                  "coin_in_usd", "theo_usd", TARGET]].copy()
    holdout["pred_log"] = pred
    holdout["pred_cipah"] = dollars(pred)
    holdout["actual_cipah"] = dollars(te[TARGET].values)
    holdout["resid"] = holdout[TARGET] - holdout["pred_log"]

    return {"model": model, "scores": scores, "holdout": holdout,
            "train_rows": len(tr), "test_rows": len(te),
            "features": FEATURES,
            "leakage_notes": [
                "Split is by calendar date, not random [2]",
                "lag7/lag28 use shift(1) so the prediction date is excluded [2]",
                "house_win and payout are never predictors — they carry luck",
                "Baselines frozen on Jan-Apr before touching May-Jun [2]",
            ]}


def adjusted_demand(panel: pd.DataFrame, model) -> pd.DataFrame:
    """
    Refit-free scoring of the whole window to get each machine's
    model-adjusted demand residual. Validation numbers above come from the
    holdout only; these residuals are the final-estimate layer [2].
    """
    d = panel.dropna(subset=FEATURES[9:]).copy()   # need lags present
    d["pred_log"] = model.predict(d[FEATURES])
    d["resid"] = d[TARGET] - d["pred_log"]

    out = (d.groupby("machine_id")
             .apply(lambda g: pd.Series({
                 "adj_demand_resid": np.average(g["resid"],
                                                weights=g["available_hours"]),
                 "n_scored_days": len(g),
                 "pred_cipah": np.average(np.expm1(g["pred_log"]),
                                          weights=g["available_hours"]),
                 "actual_cipah": np.average(np.expm1(g[TARGET]),
                                            weights=g["available_hours"]),
             }), include_groups=False)
             .reset_index())
    out["adj_demand_pct"] = out["adj_demand_resid"].rank(pct=True) * 100
    out["pred_theo_per_hr_gap"] = out["actual_cipah"] - out["pred_cipah"]
    return out


def refit_full(panel: pd.DataFrame, model_params=None):
    """After validating the method, refit on all six months for July [2]."""
    d = panel.dropna(subset=[TARGET])
    m = HistGradientBoostingRegressor(
        loss="absolute_error", max_depth=6, learning_rate=0.06, max_iter=500,
        categorical_features=[FEATURES.index(c) for c in CATS],
        random_state=0, **(model_params or {}))
    return m.fit(d[FEATURES], d[TARGET])