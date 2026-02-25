# rhodopipeline/growth.py
"""
MicroCT density-curve loading, density-weighted growth rate, and
Curve-6 automated increment detection / DTW alignment.

All path-dependent logic is driven by CONFIG['paths']['microct_base'].
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import find_peaks, savgol_filter

from dtaidistance import dtw as dtw_module

from rhodopipeline.config import CONFIG


# ---------------------------------------------------------------------------
# Low-level curve helpers
# ---------------------------------------------------------------------------

def compute_density_weighted_time(df_poststain, start_date, end_date):
    """
    Map physical distance along a microCT density curve to calendar time
    using v(x) = k / p(x)  ⟹  Δt_i ∝ p_i Δx_i.

    Parameters
    ----------
    df_poststain : pd.DataFrame
        Two-column frame with columns 'x' (µm position) and 'y' (density).
    start_date : datetime
        Start of the growth window (alizarin staining).
    end_date : datetime
        End of the growth window (microCT scan).

    Returns
    -------
    pd.DataFrame
        Columns: x, density, dt (days), DateTime, v (µm/day).
    """
    x = df_poststain['x'].values
    p = df_poststain['y'].values

    dx = np.diff(x)
    p_i = p[:-1]

    dt_raw = p_i * dx                                          # ∝ p_i Δx_i
    total_days = (end_date - start_date).days
    dt = dt_raw / dt_raw.sum() * total_days                   # Δt_i = P_i T

    cumulative_days = np.insert(np.cumsum(dt), 0, 0)
    dates = [start_date + timedelta(days=float(d)) for d in cumulative_days]

    v = dx / dt                                                # v(x_i) = Δx_i / Δt_i

    return pd.DataFrame({
        'x':        x[:-1],
        'density':  p_i,
        'dt':       dt,
        'DateTime': dates[:-1],
        'v':        v,
    })


def average_curves(dfs, n_points=1000):
    """
    Average multiple density curves onto a common x-grid by interpolation.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Each frame must have columns 'x' and 'y'.
    n_points : int
        Resolution of the common grid.

    Returns
    -------
    pd.DataFrame
        Columns: x, y (mean density), y_std (std of density).
    """
    x_common = np.linspace(
        min(df['x'].min() for df in dfs),
        max(df['x'].max() for df in dfs),
        n_points,
    )
    ys_interp = [np.interp(x_common, df['x'], df['y']) for df in dfs]
    y_mean = np.mean(ys_interp, axis=0)
    y_std  = np.std(ys_interp, axis=0)
    return pd.DataFrame({'x': x_common, 'y': y_mean, 'y_std': y_std})


# ---------------------------------------------------------------------------
# Wrapper: load all four post-stain density curves
# ---------------------------------------------------------------------------

def load_density_curves(base_path=None):
    """
    Load and preprocess the four post-stain microCT density CSV files.

    Each file stores (x, y) pairs where x is physical position (µm) and
    y is greyscale density.  Curves 1–2 are trimmed from the start; curves
    3–4 are trimmed from the end and their x-axes are reversed so that all
    four curves run in the same direction (growth tip → stain mark).

    Parameters
    ----------
    base_path : str, optional
        Directory containing the four CSV files.  Defaults to
        ``CONFIG['paths']['microct_base']``.

    Returns
    -------
    list of pd.DataFrame
        Four DataFrames, each with columns 'x' and 'y'.
    """
    if base_path is None:
        base_path = CONFIG['paths']['microct_base']

    # (filename, trim_side, n_rows_to_remove)
    specs = [
        ('post stain growth_1000.csv',   'first', 81),
        ('post stain growth_1000_2.csv', 'first', 112),
        ('post stain growth_1000_3.csv', 'last',  32),
        ('post stain growth_1000_4.csv', 'last',  19),
    ]

    curves = []
    for fname, side, n in specs:
        df = pd.read_csv(base_path + fname, header=None, names=['x', 'y'])
        if side == 'first':
            df = df.iloc[n:].reset_index(drop=True)
            df['x'] = df['x'] - df['x'].iloc[0]
        else:
            df = df.iloc[:-n].reset_index(drop=True)
            df['x'] = -df['x'] + df['x'].values[-1]
            df = df.iloc[::-1].reset_index(drop=True)
        curves.append(df)

    return curves


# ---------------------------------------------------------------------------
# Wrapper: Savitzky-Golay peak detection on Curve 6
# ---------------------------------------------------------------------------

def detect_curve6_increments(
    df_incwidth,
    remove_start=41,
    remove_end=35,
    window_length=11,
    polyorder=2,
    peak_distance=10,
    peak_prominence=0.005,
):
    """
    Detect automated growth increments from the Curve 6 density profile.

    Applies Savitzky-Golay smoothing, then locates peaks and valleys with
    ``scipy.signal.find_peaks``.  The gaps between consecutive extrema are
    returned as derived increment widths.

    Parameters
    ----------
    df_incwidth : pd.DataFrame
        Excel data frame with columns 'curve6x', 'curve6y', and 'curve6'
        (manual increments).
    remove_start, remove_end : int
        Rows to skip from the interior of the curve (index slicing).
    window_length, polyorder : int
        Savitzky-Golay filter parameters.
    peak_distance, peak_prominence : int / float
        ``find_peaks`` kwargs controlling minimum spacing and minimum
        prominence of detected extrema.

    Returns
    -------
    derived_increments : np.ndarray
        Width of each gap between consecutive extrema.
    derived_positions : np.ndarray
        x-positions of the left edge of each increment.
    """
    curve6x = df_incwidth['curve6x'].values
    curve6y = df_incwidth['curve6y'].values

    x_section_raw = (curve6x[-1] - curve6x)[-remove_start:remove_end:-1]
    x_section = x_section_raw - x_section_raw[0]
    y_section = curve6y[-remove_start:remove_end:-1]

    if len(y_section) >= window_length:
        y_smooth = savgol_filter(y_section, window_length, polyorder)
    else:
        print(
            f'Warning: Curve 6 section length ({len(y_section)}) '
            f'< window_length ({window_length}); using raw density.'
        )
        y_smooth = y_section

    peaks,   _ = find_peaks( y_smooth, distance=peak_distance, prominence=peak_prominence)
    valleys, _ = find_peaks(-y_smooth, distance=peak_distance, prominence=peak_prominence)

    ext_idx = np.sort(np.concatenate([peaks, valleys]))
    ext_idx = ext_idx[(ext_idx > 0) & (ext_idx < len(x_section) - 1)]

    if len(ext_idx) < 2:
        print('Warning: < 2 extrema found; derived increments empty.')
        return np.array([]), np.array([])

    derived_positions  = x_section[ext_idx[:-1]]
    derived_increments = np.diff(x_section[ext_idx])
    return derived_increments, derived_positions


# ---------------------------------------------------------------------------
# Wrapper: DTW alignment + OLS regression of increment sequences
# ---------------------------------------------------------------------------

def align_and_regress_increments(derived_seq, manual_seq):
    """
    Align two increment sequences with Dynamic Time Warping, then fit OLS.

    Parameters
    ----------
    derived_seq : array-like
        Automated (derived) increment widths.
    manual_seq : array-like
        Manually measured increment widths.

    Returns
    -------
    model : statsmodels RegressionResultsWrapper
        Fitted OLS model: manual ~ const + derived  (after DTW alignment).
    derived_aligned : np.ndarray
        DTW-aligned derived increments.
    manual_aligned : np.ndarray
        DTW-aligned manual increments.

    Raises
    ------
    ValueError
        If either sequence is empty after NaN removal.
    """
    derived_seq = np.asarray(derived_seq, dtype=float)
    manual_seq  = np.asarray(manual_seq,  dtype=float)
    derived_seq = derived_seq[~np.isnan(derived_seq)]
    manual_seq  = manual_seq[~np.isnan(manual_seq)]

    if len(derived_seq) == 0 or len(manual_seq) == 0:
        raise ValueError('Derived or manual sequence is empty after NaN removal.')

    path = dtw_module.warping_path(derived_seq, manual_seq)
    derived_aligned = derived_seq[[p[0] for p in path]]
    manual_aligned  = manual_seq[[p[1] for p in path]]

    mask = ~np.isnan(derived_aligned) & ~np.isnan(manual_aligned)
    X_mat = sm.add_constant(derived_aligned[mask].reshape(-1, 1))
    model = sm.OLS(manual_aligned[mask], X_mat).fit()

    return model, derived_aligned, manual_aligned
