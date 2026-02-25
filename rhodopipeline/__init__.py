# rhodopipeline/__init__.py
"""
rhodopipeline — local (non-installable) Python package for the
Rhodolith Temperature Proxy Pipeline.

Usage on Google Colab
---------------------
After mounting Drive and adding the repo root to sys.path::

    import sys
    REPO_PATH = '/content/drive/MyDrive/rhodopipeline'
    if REPO_PATH not in sys.path:
        sys.path.insert(0, REPO_PATH)

    import rhodopipeline                   # verify import
    from rhodopipeline import RhodolithPipeline, CONFIG

Import dependency order (no circular imports):
    config → utils → growth → dtw → plotting / stats → __init__
"""

from rhodopipeline.config import CONFIG, COLORS

from rhodopipeline.utils import _find_col, _reg_stats

from rhodopipeline.growth import (
    compute_density_weighted_time,
    average_curves,
    load_density_curves,
    detect_curve6_increments,
    align_and_regress_increments,
)

from rhodopipeline.dtw import RhodolithPipeline

from rhodopipeline.plotting import (
    plot_results,
    plot_individual_branch_reconstructions,
    plot_element_correlations_heatmap,
)

from rhodopipeline.stats import (
    permutation_test_mgsr,
    proxy_only_null_flat_master,
)

__all__ = [
    # Config
    'CONFIG',
    'COLORS',
    # Utils
    '_find_col',
    '_reg_stats',
    # Growth / microCT
    'compute_density_weighted_time',
    'average_curves',
    'load_density_curves',
    'detect_curve6_increments',
    'align_and_regress_increments',
    # Pipeline class
    'RhodolithPipeline',
    # Plotting
    'plot_results',
    'plot_individual_branch_reconstructions',
    'plot_element_correlations_heatmap',
    # Stats
    'permutation_test_mgsr',
    'proxy_only_null_flat_master',
]
