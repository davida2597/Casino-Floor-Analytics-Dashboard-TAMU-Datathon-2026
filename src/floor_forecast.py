# src/floor_forecast.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from .io_utils import con
from .config import TRAIN_END, HOLDOUT_BEG


def floor_panel(c=None) -> pd.DataFrame:
    c = c or con()
    return c.execute("""
        SELECT p.gaming_date, p.coin_in_usd AS floor_coin_in,
               cal.day_of_week, cal.is_holiday,
               MONTH(p.gaming_date) AS month,
               EXISTS (SELECT 1 FROM events e
                       WHERE p.gaming_date BETWEEN e.start_date AND e.end_date)
                   AS has_event
        FROM property p JOIN cal USING (gaming_date)
        ORDER BY 1
    """).df()


def validate_floor(fp: pd.DataFrame) -> dict:
    gd = pd.to_datetime(fp.gaming_date)
    tr = fp[gd <= pd.Timestamp(TRAIN_END)]
    te = fp[gd >= pd.Timestamp(HOLDOUT_BEG)]

    const = tr.floor_coin_in.mean()
    dow = tr.groupby("day_of_week").floor_coin_in.mean()
    dow_ev = tr.groupby(["day_of_week", "has_event"]).floor_coin_in.mean()

    p_const = np.full(len(te), const)
    p_dow = te.day_of_week.map(dow).fillna(const).values
    p_dow_ev = (te.set_index(["day_of_week", "has_event"]).index
                  .map(dow_ev).to_numpy(dtype=float))
    p_dow_ev = np.where(np.isnan(p_dow_ev), p_dow, p_dow_ev)

    res = pd.DataFrame({
        "model": ["constant_mean", "weekday_mean", "weekday_x_event"],
        "mae_usd": [mean_absolute_error(te.floor_coin_in, p) for p in
                    (p_const, p_dow, p_dow_ev)],
    })
    res["mape_pct"] = [
        np.mean(np.abs(te.floor_coin_in - p) / te.floor_coin_in) * 100
        for p in (p_const, p_dow, p_dow_ev)]

    preds = te[["gaming_date", "floor_coin_in", "day_of_week", "has_event"]].copy()
    preds["pred_constant"] = p_const
    preds["pred_weekday"] = p_dow
    preds["pred_weekday_event"] = p_dow_ev
    return {"scores": res, "preds": preds,
            "staffing_note": ("Weekday x event forecast drives the technician-"
                              "shift decision: compare predicted demand peaks "
                              "against fault-hour timing.")}