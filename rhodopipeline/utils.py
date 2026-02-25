# rhodopipeline/utils.py
"""
Shared helper functions used across multiple package modules.
"""

import logging

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def _find_col(df, includes_all=(), includes_any=()):
    """
    Fuzzy column-name matching.

    Returns the first column whose lower-cased, space-stripped, unit-stripped
    name satisfies:
      - contains ALL tokens in *includes_all*, AND
      - contains AT LEAST ONE token in *includes_any* (if non-empty)

    Prefers columns without a '.1' suffix (duplicate markers from gspread).
    Returns None if no match is found.
    """
    cols = list(df.columns)

    def ok(c):
        c0 = c.lower().replace(' ', '').split('(')[0]
        all_match = all(s.lower().replace(' ', '') in c0 for s in includes_all)
        any_match = (
            any(s.lower().replace(' ', '') in c0 for s in includes_any)
            if includes_any
            else True
        )
        return all_match and any_match

    hits = [c for c in cols if ok(c)]
    return next((h for h in hits if '.1' not in h), hits[0] if hits else None)


def _reg_stats(x, y, name=''):
    """
    Compute Pearson r, R², p-value, and RMSE for arrays *x* and *y*.

    Requires at least 3 finite paired observations; returns NaN dict otherwise.
    """
    x, y = np.asarray(x), np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return dict(r=np.nan, r2=np.nan, p=np.nan, rmse=np.nan, n=0)
    r, p = pearsonr(x[m], y[m])
    model = LinearRegression().fit(x[m].reshape(-1, 1), y[m])
    rmse = np.sqrt(mean_squared_error(y[m], model.predict(x[m].reshape(-1, 1))))
    logging.info(
        f'{name}: R={r:.3f}, R²={r**2:.3f}, p={p:.3f}, RMSE={rmse:.3f}°C (n={m.sum()})'
    )
    return dict(r=float(r), r2=float(r**2), p=float(p), rmse=float(rmse), n=int(m.sum()))
