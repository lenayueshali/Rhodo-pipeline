# rhodopipeline/plotting.py
"""
Standalone plotting functions for the Rhodolith pipeline.

Each function accepts a *pipeline* instance (RhodolithPipeline) and reads
its public attributes.  They are called from the thin delegation wrappers
in ``rhodopipeline.dtw.RhodolithPipeline``.
"""

import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from dtaidistance import dtw

from rhodopipeline.config import COLORS
from rhodopipeline.utils import _find_col, _reg_stats


def plot_results(pipeline):
    """
    Four-panel composite figure.

    Panels A–C: aligned proxy series (Mg/Sr, Mg/Ca, Sr/Ca) with temperature
    on the right axis.
    Panel D: temperature reconstructions from all three proxies against
    the logger record.
    """
    print('=' * 60)
    print('STEP 7: PLOTTING')
    print('=' * 60)

    proxies = ['Mg/Sr', 'Mg/Ca', 'Sr/Ca']
    fig, axes = plt.subplots(4, 1, figsize=(16, 22), sharex=True)

    temp_std = pipeline.temp_full.rolling(window=7, center=True).std()
    temp_std = temp_std.ffill().bfill()

    df_mgsr  = pipeline.aligned_data['Mg/Sr']
    branches = sorted(df_mgsr['Branch'].unique())
    b_colors = {b: plt.cm.tab10(i / max(1, len(branches) - 1)) for i, b in enumerate(branches)}

    calib_start = pd.to_datetime(pipeline.config['dates']['deploy_start'])

    def add_bg(ax):
        ax2 = ax.twinx()
        ax2.plot(pipeline.temp_full.index, pipeline.temp_full, 'k:', alpha=0.4, lw=1.2,
                 label='Logger mean')
        upper = pipeline.temp_full + temp_std
        lower = pipeline.temp_full - temp_std
        ax2.fill_between(pipeline.temp_full.index, lower, upper,
                         color='lightgray', alpha=0.25, label='Logger ±1σ')
        ax2.set_ylabel('Temperature (°C)', fontweight='bold', color='dimgray', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='dimgray', labelsize=12)

    # Panels A–C
    for i, proxy in enumerate(proxies):
        ax = axes[i]
        add_bg(ax)

        if proxy not in pipeline.aligned_data or proxy not in pipeline.composite_data:
            continue

        df   = pipeline.aligned_data[proxy]
        df_c = df[df['Date'] >= calib_start]

        for b, g in df_c.groupby('Branch'):
            g_sorted = g.sort_values('Date')
            ax.plot(g_sorted['Date'], g_sorted['Value'], 'o-',
                    color=b_colors[b], markersize=3, alpha=0.5, lw=0.8,
                    label=b if i == 0 else '')

        comp = pipeline.composite_data[proxy]
        comp = comp[comp.index >= calib_start]
        ax.plot(comp.index, comp['mean'], 'k-', lw=3, alpha=0.9,
                label='Composite mean' if i == 0 else '')
        ax.fill_between(comp.index,
                        comp['mean'] - comp['std'],
                        comp['mean'] + comp['std'],
                        color=COLORS.get(proxy, 'k'), alpha=0.25)

        centroids  = df_c.groupby('Branch')['Value'].mean()
        x_centroid = comp.index[-1] + pd.Timedelta(days=3)
        for b, v in centroids.items():
            ax.scatter(x_centroid, v, color=b_colors[b], edgecolor='k', s=45, zorder=5)

        ax.set_ylabel(proxy, fontweight='bold', color=COLORS.get(proxy, 'k'), fontsize=16)
        ax.tick_params(axis='y', labelsize=12)
        ax.text(0.02, 0.9, chr(65 + i), transform=ax.transAxes,
                fontweight='bold', fontsize=18,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

        if i == 0:
            ax.legend(loc='upper center', ncol=min(4, len(branches) + 1),
                      bbox_to_anchor=(0.5, 1.22), fontsize=11, framealpha=0.9)

    # Panel D: Reconstructions
    ax = axes[3]
    ax.fill_between(pipeline.temp_full.index, pipeline.temp_min, pipeline.temp_max,
                    color='gray', alpha=0.15, label='In-situ Daily Range')
    ax.plot(pipeline.temp_full.index, pipeline.temp_full, 'k--', lw=1.5, alpha=0.5,
            label='Logger Daily Mean')

    for proxy in proxies:
        if proxy not in pipeline.final_equations or proxy not in pipeline.composite_data:
            continue

        comp = pipeline.composite_data[proxy]
        comp = comp[comp.index >= calib_start]
        eq   = pipeline.final_equations[proxy]

        rec_mean = comp['mean'].values * eq['slope'] + eq['intercept']
        color    = COLORS.get(proxy, 'k')
        ax.plot(comp.index, rec_mean, 'o-', markersize=4, lw=2.0,
                alpha=0.9, color=color, label=f'{proxy} Rec')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax.grid(False)
    ax.set_ylabel('Temperature (°C)', fontweight='bold', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper left', fontsize=11, framealpha=1.0,
              edgecolor='black', fancybox=False, ncol=1)
    ax.text(0.02, 0.9, 'D', transform=ax.transAxes,
            fontweight='bold', fontsize=18,
            bbox=dict(facecolor='white', edgecolor='black', pad=4.0))

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=0, ha='center', fontsize=12)

    lines = []
    for proxy in proxies:
        if proxy in pipeline.final_equations:
            s = pipeline.final_equations[proxy]['stats']
            lines.append(f"{proxy}: $R^2$={s['r2']:.2f}, RMSE={s['rmse']:.2f}°C")
    if lines:
        fig.text(0.5, 0.03, '  |  '.join(lines), ha='center',
                 bbox=dict(facecolor='white', edgecolor='black',
                           boxstyle='square', pad=0.6),
                 fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.95)
    plt.show()


def plot_individual_branch_reconstructions(pipeline):
    """
    Per-branch validation subplots: reconstructed temperature from Mg/Sr
    overlaid on the logger record with R² and RMSE annotations.
    """
    print('\n' + '=' * 70)
    print('STEP 8: INDIVIDUAL BRANCH VALIDATION (Mg/Sr)')
    print('=' * 70)

    proxy = 'Mg/Sr'
    if proxy not in pipeline.aligned_data or proxy not in pipeline.final_equations:
        print('❌ Mg/Sr data or equation missing.')
        return

    df       = pipeline.aligned_data[proxy]
    eq       = pipeline.final_equations[proxy]
    branches = sorted(df['Branch'].unique())

    fig, axes = plt.subplots(len(branches), 1, figsize=(12, 20),
                             sharex=True, sharey=True)
    if len(branches) == 1:
        axes = [axes]

    for i, branch in enumerate(branches):
        ax           = axes[i]
        branch_data  = df[df['Branch'] == branch].sort_values('Date')
        temp_rec     = branch_data['Value'].values * eq['slope'] + eq['intercept']
        dates        = branch_data['Date']
        logger_at    = pipeline.temp_full.reindex(dates, method='nearest').values
        stats        = _reg_stats(temp_rec, logger_at, f'{branch}')

        ax.fill_between(pipeline.temp_full.index, pipeline.temp_min, pipeline.temp_max,
                        color='gray', alpha=0.15, label='In-situ Range')
        ax.plot(pipeline.temp_full.index, pipeline.temp_full, 'k--', lw=1.2, alpha=0.5)

        color = COLORS.get('Mg/Sr', 'purple')
        ax.plot(dates, temp_rec, '-', lw=1.2, color=color, alpha=0.6)
        ax.scatter(dates, temp_rec, s=25, color=color, edgecolors='white',
                   linewidth=0.6, zorder=3, label=f'{branch} Data')

        stat_text = f"{branch}: $R^2$={stats['r2']:.2f}, RMSE={stats['rmse']:.2f}°C"
        ax.text(0.02, 0.88, stat_text, transform=ax.transAxes,
                fontweight='bold', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'))

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)
        ax.set_ylabel('Temp (°C)', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        if i == 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=1.0, edgecolor='black')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=0, ha='center', fontsize=12)
    plt.xlabel('Date (2024)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.show()


def plot_element_correlations_heatmap(pipeline):
    """
    Seaborn heatmap of Pearson R (proxy element vs temperature) for all
    branches, with LaTeX-formatted isotope labels on the y-axis.

    Uses the Mg/Sr DTW alignment from the pipeline's synthetic master.
    """
    print('\n' + '=' * 70)
    print('STEP 9: MULTI-ELEMENT CORRELATION HEATMAP (Formatted Labels)')
    print('=' * 70)

    if pipeline.synthetic_master is None:
        print('❌ Synthetic master missing.')
        return

    master_vals  = pipeline.synthetic_master['MgSr_Target'].values
    master_dates = pipeline.synthetic_master['Date'].values
    window = int(pipeline.config.get('dtw', {}).get('window', 10) or 10)
    psi    = pipeline.config['dtw']['psi']

    def format_isotope_label(label):
        label = label.replace('Mean ', '').strip()
        label = re.sub(r'\(.*\)', '', label).strip()
        parts = label.split('/')
        formatted_parts = []
        for part in parts:
            m = re.match(r'([A-Za-z]+)(\d+)', part)
            if m:
                elem, mass = m.groups()
                formatted_parts.append(f'$^{{{mass}}}\\mathrm{{{elem}}}$')
            else:
                formatted_parts.append(f'$\\mathrm{{{part}}}$')
        return '/'.join(formatted_parts)

    results = []

    for i in range(1, 8):
        df_raw, name = pipeline.get_raw_branch_data(i)
        if df_raw is None:
            continue
        df = df_raw.apply(pd.to_numeric, errors='coerce')

        sr_col = _find_col(df, ('ca',), ('sr',))
        mg_col = _find_col(df, ('ca',), ('mg',))
        if not sr_col or not mg_col:
            continue

        mgsr_raw   = df[mg_col] / df[sr_col]
        valid_mask = mgsr_raw.notna()
        mgsr_clean = mgsr_raw[valid_mask].values

        try:
            path  = dtw.warping_path(mgsr_clean, master_vals, window=window, psi=psi)
            q_idx = [p[0] for p in path]
            m_idx = [p[1] for p in path]

            aligned_dates = master_dates[m_idx]
            aligned_temps = pipeline.temp_full.reindex(aligned_dates, method='nearest').values

            ratio_cols = [c for c in df.columns if '/' in c and 'mmol' in c]
            df_clean   = df.loc[valid_mask].iloc[q_idx]

            for col in ratio_cols:
                vals      = df_clean[col].values
                mask_corr = np.isfinite(vals) & np.isfinite(aligned_temps)

                if mask_corr.sum() > 20 and np.std(vals[mask_corr]) > 0:
                    r, p = pearsonr(vals[mask_corr], aligned_temps[mask_corr])
                    results.append({
                        'Branch':        name,
                        'Element_Ratio': col,
                        'Correlation':   r,
                        'p_value':       p,
                    })
        except Exception:
            continue

    if not results:
        print('❌ No correlations found.')
        return

    res_df = pd.DataFrame(results)
    res_df['Formatted_Label'] = res_df['Element_Ratio'].apply(format_isotope_label)

    grouped = (
        res_df
        .groupby(['Branch', 'Formatted_Label'])
        .agg({'Correlation': 'mean', 'p_value': 'mean'})
        .reset_index()
    )
    grouped['annot'] = grouped.apply(
        lambda row: f"R = {row['Correlation']:.2f}\n($p$ = {row['p_value']:.3f})",
        axis=1,
    )

    heatmap_r     = grouped.pivot(index='Formatted_Label', columns='Branch', values='Correlation')
    heatmap_annot = grouped.pivot(index='Formatted_Label', columns='Branch', values='annot')

    mean_corr     = heatmap_r.mean(axis=1).sort_values(ascending=False)
    heatmap_r     = heatmap_r.reindex(mean_corr.index)
    heatmap_annot = heatmap_annot.reindex(mean_corr.index)

    plt.figure(figsize=(10, len(heatmap_r) * 0.6 + 2))
    ax = sns.heatmap(
        heatmap_r,
        annot=heatmap_annot,
        fmt='',
        center=0, vmin=-1, vmax=1, cmap='RdBu_r',
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Pearson Correlation (R)'},
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.title(
        'Correlation of Element Ratios with Temperature\n(Aligned via Mg/Sr)',
        fontweight='bold', pad=15,
    )
    plt.xlabel(None)
    plt.ylabel(None)
    plt.tight_layout()
    plt.show()
