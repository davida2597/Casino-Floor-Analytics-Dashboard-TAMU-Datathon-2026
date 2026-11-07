# src/payout.py
import numpy as np
import pandas as pd
from scipy import stats


def _bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values; NaN p-values stay NaN and are excluded from m."""
    p = pd.Series(p, dtype=float)
    q = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    m = int(ok.sum())
    if m == 0:
        return q
    order = p[ok].sort_values().index
    ranked = p[order].to_numpy() * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # enforce monotonicity
    q.loc[order] = np.clip(ranked, 0, 1)
    return q


def luck_test(sc: pd.DataFrame) -> pd.DataFrame:
    """
    Test observed hold against configured expected hold, using the paytable's
    published payout-multiplier variance. SE(house win) = sqrt(C * b * var),
    where C is total coin-in and b the average bet [2].
    """
    out = sc.copy()

    var = out.get("payout_var")
    if var is None or var.isna().all():
        out["luck_var_source"] = "defaulted (var=9)"
        var = pd.Series(9.0, index=out.index)
    else:
        out["luck_var_source"] = "game_math.payout_multiplier_variance"
        var = var.fillna(var.median())

    bet = out.get("avg_bet_usd")
    bet = (pd.Series(1.50, index=out.index) if bet is None
           else bet.fillna(bet.median() if bet.notna().any() else 1.50))

    C = out["coin_in"].clip(lower=0)
    out["luck_se"] = np.sqrt(C * bet * var)
    out["luck_z"] = out["luck_residual"] / out["luck_se"].replace(0, np.nan)

    # Benjamini-Hochberg across ~310 machines: at p<0.05 uncorrected you
    # would "find" roughly 15 suspicious machines by chance alone [2].
    out["luck_p"] = 2 * stats.norm.sf(out["luck_z"].abs())
    out["luck_q"] = _bh(out["luck_p"])

    out["payout_flag"] = np.select(
        [out["luck_q"] < 0.05, out["luck_q"] < 0.20],
        ["inspect", "watch"], default="normal")
    out.loc[out["low_exposure"], "payout_flag"] = "too_little_data"

    se_hold = out["luck_se"] / C.replace(0, np.nan)
    out["hold_lo"] = out["observed_hold"] - 1.96 * se_hold
    out["hold_hi"] = out["observed_hold"] + 1.96 * se_hold
    out["hold_ci_covers_expected"] = (
        (out["hold_lo"] <= out["expected_hold"]) &
        (out["expected_hold"] <= out["hold_hi"]))

    # Detectable-effect floor: how big a hold deviation could this exposure
    # even reveal? Machines with a wide band cannot be judged on hold at all.
    out["min_detectable_hold_pp"] = 1.96 * se_hold * 100
    return out