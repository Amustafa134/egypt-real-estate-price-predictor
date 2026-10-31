"""
Evaluation metrics for the Egypt Real Estate Price Predictor.

Models predict log(price). This module reports errors both on the log scale
(what the model optimises) and back in EGP (what a business user cares about).
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true_log, y_pred_log) -> dict:
    """
    Compare true and predicted log(price) values.

    Returns:
        rmse_log, mae_log, r2_log      -> errors on the log scale
        rmse_egp, mae_egp, r2_egp      -> errors back in EGP
        median_abs_pct_error           -> typical % error (0.15 = 15%)
        within_20pct                   -> share of listings predicted
                                          within +/-20% of the true price

    Note: exp(predicted log price) estimates the median price, not the mean,
    so EGP-scale errors are slightly conservative for the expensive tail.
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)

    true_egp = np.exp(y_true_log)
    pred_egp = np.exp(y_pred_log)
    abs_pct_error = np.abs(pred_egp - true_egp) / true_egp

    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "rmse_egp": float(np.sqrt(mean_squared_error(true_egp, pred_egp))),
        "mae_egp": float(mean_absolute_error(true_egp, pred_egp)),
        "r2_egp": float(r2_score(true_egp, pred_egp)),
        "median_abs_pct_error": float(np.median(abs_pct_error)),
        "within_20pct": float(np.mean(abs_pct_error <= 0.20)),
    }
