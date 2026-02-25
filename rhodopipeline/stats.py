# rhodopipeline/stats.py
"""
Statistical validation functions for the Rhodolith pipeline.

Both functions accept a *pipeline* instance (RhodolithPipeline) and access
its ``composite_data``, ``temp_full``, and ``synthetic_master`` attributes.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def permutation_test_mgsr(pipeline, n_perm=200, block_len=7, random_state=42):
    """
    Permutation test for the Mg/Sr–temperature calibration.

    Permutes the logger temperature record relative to the Mg/Sr composite
    while preserving autocorrelation, refits T ~ Mg/Sr, and records R² for
    each permutation.  Two permutation schemes are used:

    - **circular shifts** — shifts the temperature array by a random offset;
    - **block permutations** — shuffles contiguous blocks of *block_len* days.

    Parameters
    ----------
    pipeline : RhodolithPipeline
        Must have ``composite_data['Mg/Sr']`` and ``temp_full`` populated.
    n_perm : int
        Number of permutations per scheme.
    block_len : int
        Number of days per block for the block-permutation scheme.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``observed_r2``, ``shift_r2`` (array), ``block_r2`` (array).
    """
    rng = np.random.default_rng(random_state)

    comp         = pipeline.composite_data['Mg/Sr']
    common_dates = comp.index.intersection(pipeline.temp_full.index)
    x = comp.loc[common_dates, 'mean'].values
    y = pipeline.temp_full.loc[common_dates].values
    n = len(x)

    # In-sample R² for reference
    msk     = np.isfinite(x) & np.isfinite(y)
    X       = x[msk].reshape(-1, 1)
    y_clean = y[msk]
    base_model = LinearRegression().fit(X, y_clean)
    base_r2    = r2_score(y_clean, base_model.predict(X))
    print(f'Observed Mg/Sr R² = {base_r2:.3f} (n={len(y_clean)})')

    def _fit_r2(y_perm):
        m = np.isfinite(x) & np.isfinite(y_perm)
        if m.sum() < 10:
            return np.nan
        Xp = x[m].reshape(-1, 1)
        yp = y_perm[m]
        model = LinearRegression().fit(Xp, yp)
        return r2_score(yp, model.predict(Xp))

    # 1a. Circular shifts
    perm_r2_shift = []
    for _ in range(n_perm):
        k = rng.integers(1, n)
        perm_r2_shift.append(_fit_r2(np.roll(y, k)))

    # 1b. Block permutations
    perm_r2_block = []
    starts = np.arange(0, n, block_len)
    for _ in range(n_perm):
        blocks = [slice(s, min(s + block_len, n)) for s in starts]
        rng.shuffle(blocks)
        idx     = np.concatenate([np.arange(b.start, b.stop) for b in blocks])
        perm_r2_block.append(_fit_r2(y[idx]))

    perm_r2_shift = np.array(perm_r2_shift)
    perm_r2_block = np.array(perm_r2_block)

    print('\nPermutation test (circular shifts):')
    print(f'  Median null R²         = {np.nanmedian(perm_r2_shift):.3f}')
    print(f'  95th percentile null R² = {np.nanpercentile(perm_r2_shift, 95):.3f}')
    print(f'  Proportion null R² >= observed = {np.mean(perm_r2_shift >= base_r2):.3f}')

    print('\nPermutation test (block permutations):')
    print(f'  Median null R²         = {np.nanmedian(perm_r2_block):.3f}')
    print(f'  95th percentile null R² = {np.nanpercentile(perm_r2_block, 95):.3f}')
    print(f'  Proportion null R² >= observed = {np.mean(perm_r2_block >= base_r2):.3f}')

    return {
        'observed_r2': base_r2,
        'shift_r2':    perm_r2_shift,
        'block_r2':    perm_r2_block,
    }


def proxy_only_null_flat_master(pipeline, window_days=None):
    """
    Null-model test using a flat (constant) synthetic master.

    Replaces the temperature-based Mg/Sr master with a constant equal to
    the mean of the current master, re-runs DTW alignment and calibration,
    and returns the Mg/Sr R² under this null model.  The original master is
    restored after the test.

    Parameters
    ----------
    pipeline : RhodolithPipeline
        Must have ``synthetic_master`` populated and ``perform_dtw_alignment``
        / ``build_composite_and_calibrate`` available.
    window_days : int or None
        Sakoe–Chiba window to use for the null-model DTW run.

    Returns
    -------
    float
        Mg/Sr R² under the flat-master null model.
    """
    master_backup = pipeline.synthetic_master.copy()

    mean_mgsr  = master_backup['MgSr_Target'].mean()
    flat_master = master_backup.copy()
    flat_master['MgSr_Target'] = mean_mgsr
    pipeline.synthetic_master  = flat_master

    print('\n=== PROXY-ONLY NULL: FLAT MASTER ===')
    pipeline.perform_dtw_alignment(window_days=window_days)
    pipeline.build_composite_and_calibrate()

    r2_null = pipeline.final_equations['Mg/Sr']['stats']['r2']
    print(f'Flat-master Mg/Sr R² = {r2_null:.3f}')

    pipeline.synthetic_master = master_backup
    return r2_null
