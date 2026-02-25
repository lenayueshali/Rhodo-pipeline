# rhodopipeline/dtw.py
"""
RhodolithPipeline — full DTW-based temperature-proxy reconstruction pipeline.

Workflow
--------
1. authenticate()                     — mount Drive, authorise gspread
2. load_temperature_data()            — logger CSV → daily mean/max/min
3. load_curve6_curve7()               — microCT time axes from Rhodo25_data
4. screen_best_linear_branch()        — pick best Mg/Sr–temperature branch
5. generate_synthetic_master()        — predicted Mg/Sr series from temperature
6. perform_dtw_alignment(window_days) — warp each branch onto master
7. build_composite_and_calibrate()    — composite Mg/Sr → T calibration
8. block_bootstrap_calibration()      — 95 % CI on slope/intercept/RMSE
9. screen_all_elements_per_branch()   — multi-element screening table
10. plot_*()                          — delegate to rhodopipeline.plotting
"""

import logging
import re
import warnings

import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from google.colab import auth, drive
from google.auth import default
import gspread

from rhodopipeline.config import CONFIG, COLORS
from rhodopipeline.utils import _find_col, _reg_stats

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


class RhodolithPipeline:
    """
    Rhodolith Temperature Proxy Pipeline using Mg/Sr-based DTW alignment.
    """

    def __init__(self, config):
        self.config = config
        self.gc = None

        # Environmental
        self.temp_full  = None
        self.temp_max   = None
        self.temp_min   = None
        self.full_series = None

        # Screening
        self.best_branch_meta = {}

        # Synthetic master
        self.synthetic_master = None

        # Alignment
        self.aligned_data     = {}
        self.dtw_diagnostics  = []

        # Composite & calibration
        self.composite_data  = {}
        self.final_equations = {}

        # Diagnostics
        self.bootstrap_results  = {}
        self.leave_one_out_stats = None

        # Curve6/7
        self.curve6 = None
        self.curve7 = None

    # ------------------------------------------------------------------
    # STEP 1: AUTHENTICATION
    # ------------------------------------------------------------------

    def authenticate(self):
        print('=' * 60)
        print('STEP 1: AUTHENTICATION')
        print('=' * 60)
        drive.mount('/content/drive', force_remount=True)
        auth.authenticate_user()
        creds, _ = default()
        self.gc = gspread.authorize(creds)
        print('✓ Authenticated.\n')

    # ------------------------------------------------------------------
    # STEP 2: TEMPERATURE
    # ------------------------------------------------------------------

    def load_temperature_data(self):
        print('=' * 60)
        print('STEP 2: LOAD TEMPERATURE')
        print('=' * 60)

        df = pd.read_csv(self.config['paths']['wally_csv'], parse_dates=['Index'])
        mask = (
            (df['Index'] >= self.config['dates']['deploy_start']) &
            (df['Index'] <= self.config['dates']['deploy_end'])
        )
        full_series = df.loc[mask].set_index('Index')['temp.exposed.af']

        self.full_series = full_series
        self.temp_full   = full_series.resample('D').mean().interpolate('time')
        self.temp_max    = full_series.resample('D').max().interpolate('time')
        self.temp_min    = full_series.resample('D').min().interpolate(method='time')

        print(f'✓ Temp loaded: {len(self.temp_full)} days.')
        print(f'  Range: {self.temp_full.min():.2f}–{self.temp_full.max():.2f} °C\n')

    # ------------------------------------------------------------------
    # UTIL: RAW BRANCH DATA
    # ------------------------------------------------------------------

    def get_raw_branch_data(self, branch_num):
        target_name_lower = f'afe5-{branch_num}'.lower()
        try:
            wb = self.gc.open(self.config['sheets']['laser_workbook'])
            found_ws = None
            for ws in wb.worksheets():
                if target_name_lower in ws.title.lower().replace(' ', ''):
                    found_ws = ws
                    break
            if found_ws:
                raw = found_ws.get_all_values()
                if len(raw) > 1:
                    df = pd.DataFrame(raw[1:], columns=raw[0])
                    df = df.loc[:, ~df.columns.duplicated(keep='first')]
                    clean_name = re.sub(r'(?i)copy of|raw', '', found_ws.title).strip()
                    return df, clean_name
        except Exception as e:
            logging.error(f'Error accessing branch {branch_num}: {e}')
        return None, None

    # ------------------------------------------------------------------
    # CURVE 6 & CURVE 7 LOAD
    # ------------------------------------------------------------------

    def load_curve6_curve7(self):
        print('=' * 60)
        print('LOAD CURVE6 & CURVE7 (Rhodo25_data)')
        print('=' * 60)
        try:
            wb = self.gc.open('Rhodo25_data')
        except Exception as e:
            print(f'❌ Could not open Rhodo25_data: {e}')
            self.curve6 = None
            self.curve7 = None
            return

        def _load_curve(sheet_name):
            try:
                ws  = wb.worksheet(sheet_name)
                raw = ws.get_all_values()
                headers = [h.strip() for h in raw[0]]
                data = []
                for row in raw[1:]:
                    row_data = {}
                    for i, h in enumerate(headers):
                        if i < len(row):
                            row_data[h] = row[i]
                    data.append(row_data)
                df = pd.DataFrame(data)
                df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
                df = df.dropna(subset=['DateTime']).sort_values('DateTime')
                return df
            except Exception as e:
                print(f"  ⚠ Error loading '{sheet_name}': {e}")
                return None

        self.curve6 = _load_curve('curve6')
        self.curve7 = _load_curve('curve7')

        if self.curve6 is not None:
            print(f'✓ Curve6 loaded ({len(self.curve6)} increments)')
        if self.curve7 is not None:
            print(f'✓ Curve7 loaded ({len(self.curve7)} increments)')
        print()

    # ------------------------------------------------------------------
    # HELPER: TIME AXIS FROM CURVE
    # ------------------------------------------------------------------

    def _time_axis_from_curve(self, n_points, curve_df):
        curve_times = curve_df['DateTime'].dropna().sort_values().values
        if len(curve_times) < 2:
            t_start = pd.to_datetime(self.config['dates']['deploy_start'])
            t_end   = pd.to_datetime(self.config['dates']['deploy_end'])
            return pd.date_range(start=t_start, end=t_end, periods=n_points)

        t_param   = np.linspace(0, 1, len(curve_times))
        t_new     = np.linspace(0, 1, n_points)
        ordinals  = curve_times.astype('datetime64[ns]').astype('int64')
        ord_interp = np.interp(t_new, t_param, ordinals)
        return pd.to_datetime(ord_interp)

    # ------------------------------------------------------------------
    # STEP 3: SCREENING
    # ------------------------------------------------------------------

    def screen_best_linear_branch(self):
        """
        Fit linear Mg/Sr ~ Temperature models for all branches plus
        Curve6/Curve7 time axes for AFE5-1.  Stores the winner in
        ``self.best_branch_meta``.
        """
        print('=' * 60)
        print('STEP 3: SCREENING (Linear + Curve6/7 for AFE5-1)')
        print('=' * 60)

        t_start = pd.to_datetime(self.config['dates']['deploy_start'])
        t_end   = pd.to_datetime(self.config['dates']['deploy_end'])

        best_r2         = -np.inf
        best_branch     = None
        best_model_type = None
        best_params     = {}

        for i in range(1, 8):
            df_raw, name = self.get_raw_branch_data(i)
            if df_raw is None:
                continue

            df     = df_raw.apply(pd.to_numeric, errors='coerce')
            sr_col = _find_col(df, ('ca',), ('sr',))
            mg_col = _find_col(df, ('ca',), ('mg',))
            if not sr_col or not mg_col:
                continue

            mgsr  = df[mg_col] / df[sr_col]
            valid = mgsr.dropna()
            if len(valid) < self.config['min_spots']:
                continue

            y = valid.values

            # (A) Linear time axis
            linear_dates = pd.date_range(start=t_start, end=t_end, periods=len(valid))
            temps_lin    = self.temp_full.reindex(linear_dates, method='nearest').values
            X_lin  = temps_lin.reshape(-1, 1)
            mask_lin = np.isfinite(X_lin.ravel()) & np.isfinite(y)
            r2_lin = np.nan
            slope_lin = np.nan
            intercept_lin = np.nan
            if mask_lin.sum() >= 10:
                model_lin     = LinearRegression().fit(X_lin[mask_lin], y[mask_lin])
                r2_lin        = model_lin.score(X_lin[mask_lin], y[mask_lin])
                slope_lin     = float(model_lin.coef_[0])
                intercept_lin = float(model_lin.intercept_)
            print(f'  {name}: Linear R² = {r2_lin:.4f}')

            if r2_lin > best_r2:
                best_r2         = r2_lin
                best_branch     = name
                best_model_type = 'linear'
                best_params     = {'slope': slope_lin, 'intercept': intercept_lin}

            # (B) Curve6/7 time axis for AFE5-1
            if i == 1:
                for mtype, curve_attr in [('curve6', 'curve6'), ('curve7', 'curve7')]:
                    curve_df = getattr(self, curve_attr, None)
                    if curve_df is None:
                        continue
                    dates_curve  = self._time_axis_from_curve(len(valid), curve_df)
                    temps_curve  = self.temp_full.reindex(dates_curve, method='nearest').values
                    X_curve      = temps_curve.reshape(-1, 1)
                    mask_curve   = np.isfinite(X_curve.ravel()) & np.isfinite(y)
                    r2_curve     = np.nan
                    slope_curve  = np.nan
                    intercept_curve = np.nan
                    if mask_curve.sum() >= 10:
                        model_curve     = LinearRegression().fit(X_curve[mask_curve], y[mask_curve])
                        r2_curve        = model_curve.score(X_curve[mask_curve], y[mask_curve])
                        slope_curve     = float(model_curve.coef_[0])
                        intercept_curve = float(model_curve.intercept_)
                    print(f'      {mtype} time R² = {r2_curve:.4f}')

                    if r2_curve > best_r2:
                        best_r2         = r2_curve
                        best_branch     = name
                        best_model_type = mtype
                        best_params     = {'slope': slope_curve, 'intercept': intercept_curve}

        if best_branch is None:
            print('❌ Screening failed (no valid branches).\n')
            return

        self.best_branch_meta = best_params
        self.best_branch_meta['name']       = best_branch
        self.best_branch_meta['model_type'] = best_model_type

        print(f'\n✓ WINNER: {best_branch} using {best_model_type} time axis')
        print(f'  Mg/Sr = {best_params["slope"]:.4f} * Temp + {best_params["intercept"]:.4f}')
        print(f'  Best R² = {best_r2:.4f}\n')

    # ------------------------------------------------------------------
    # STEP 4: SYNTHETIC MASTER
    # ------------------------------------------------------------------

    def generate_synthetic_master(self):
        print('=' * 60)
        print('STEP 4: GENERATE SYNTHETIC Mg/Sr MASTER')
        print('=' * 60)

        if not self.best_branch_meta:
            print('❌ Run screen_best_linear_branch() first.\n')
            return

        m = self.best_branch_meta['slope']
        c = self.best_branch_meta['intercept']
        pred_mgsr = m * self.temp_full + c
        self.synthetic_master = pd.DataFrame({
            'Date':       self.temp_full.index,
            'MgSr_Target': pred_mgsr,
        })

        print(f'✓ Synthetic Master Created ({len(pred_mgsr)} days)')
        print(f'  Mg/Sr range: {pred_mgsr.min():.2f}–{pred_mgsr.max():.2f} mmol/mol\n')

    # ------------------------------------------------------------------
    # STEP 5: DTW ALIGNMENT
    # ------------------------------------------------------------------

    def perform_dtw_alignment(self, window_days=15, label_suffix=''):
        print('=' * 60)
        print(f'STEP 5: UNIFIED DTW ALIGNMENT (window={window_days} days)')
        print('=' * 60)

        if self.synthetic_master is None:
            print('❌ Synthetic master missing; run generate_synthetic_master() first.\n')
            return

        master_vals  = self.synthetic_master['MgSr_Target'].values
        master_dates = self.synthetic_master['Date'].values
        window       = int(window_days) if window_days is not None else None

        aligned_mgsr_list = []
        aligned_mgca_list = []
        aligned_srca_list = []
        self.dtw_diagnostics = []

        for i in range(1, 8):
            df_raw, name = self.get_raw_branch_data(i)
            if df_raw is None:
                continue

            df     = df_raw.apply(pd.to_numeric, errors='coerce')
            sr_col = _find_col(df, ('ca',), ('sr',))
            mg_col = _find_col(df, ('ca',), ('mg',))
            if not sr_col or not mg_col:
                logging.warning(f'{name}: missing Mg or Sr column, skipping')
                continue

            mg   = df[mg_col]
            sr   = df[sr_col]
            mgsr = mg / sr

            valid_mask = mgsr.notna() & mg.notna() & sr.notna()
            if valid_mask.sum() < self.config['min_spots']:
                logging.warning(f'{name}: only {valid_mask.sum()} valid points, skipping')
                continue

            mgsr_clean = mgsr[valid_mask].values
            mg_clean   = mg[valid_mask].values
            sr_clean   = sr[valid_mask].values

            try:
                path = dtw.warping_path(
                    mgsr_clean, master_vals,
                    window=window,
                    psi=self.config['dtw']['psi'],
                )

                n_q     = len(mgsr_clean)
                n_m     = len(master_vals)
                stretch = len(path) / max(n_q, n_m)
                self.dtw_diagnostics.append({
                    'branch':         name,
                    'n_query':        n_q,
                    'n_master':       n_m,
                    'path_len':       len(path),
                    'stretch_factor': stretch,
                })

                q_idx = [p[0] for p in path]
                m_idx = [p[1] for p in path]
                dates = master_dates[m_idx]

                aligned_mgsr_list.append(pd.DataFrame({'Date': dates, 'Branch': name, 'Value': mgsr_clean[q_idx]}))
                aligned_mgca_list.append(pd.DataFrame({'Date': dates, 'Branch': name, 'Value': mg_clean[q_idx]}))
                aligned_srca_list.append(pd.DataFrame({'Date': dates, 'Branch': name, 'Value': sr_clean[q_idx]}))

                print(f'  ✓ Aligned {name}: {len(path)} steps, stretch={stretch:.2f}')
            except Exception as e:
                logging.error(f'DTW failed for {name}: {e}')

        if not aligned_mgsr_list:
            print('❌ No branches aligned; check DTW settings.\n')
            return

        self.aligned_data['Mg/Sr'] = pd.concat(aligned_mgsr_list, ignore_index=True)
        self.aligned_data['Mg/Ca'] = pd.concat(aligned_mgca_list, ignore_index=True)
        self.aligned_data['Sr/Ca'] = pd.concat(aligned_srca_list, ignore_index=True)

        diag_df = pd.DataFrame(self.dtw_diagnostics)
        print('\nDTW stretch factors:')
        print(diag_df[['branch', 'stretch_factor']].to_string(index=False))
        print(f'\nMean stretch factor: {diag_df["stretch_factor"].mean():.2f}\n')

    # ------------------------------------------------------------------
    # STEP 6: COMPOSITE & CALIBRATION
    # ------------------------------------------------------------------

    def build_composite_and_calibrate(self):
        print('=' * 60)
        print('STEP 6: COMPOSITE & FINAL CALIBRATION')
        print('=' * 60)

        calib_start = pd.to_datetime(self.config['dates']['deploy_start'])

        self.composite_data  = {}
        self.final_equations = {}

        for base in ['Mg/Sr', 'Mg/Ca', 'Sr/Ca']:
            if base not in self.aligned_data:
                continue

            df  = self.aligned_data[base]
            df  = df[df['Date'] >= calib_start]

            comp = df.groupby('Date')['Value'].agg(['mean', 'std', 'count']).sort_index()
            self.composite_data[base] = comp

            common_dates = comp.index.intersection(self.temp_full.index)
            if len(common_dates) < 10:
                print(f'  ⚠ {base}: too few overlapping days for calibration.')
                continue

            x_proxy = comp.loc[common_dates, 'mean'].values
            y_temp  = self.temp_full.loc[common_dates].values
            mask    = np.isfinite(x_proxy) & np.isfinite(y_temp)
            if mask.sum() < 10:
                print(f'  ⚠ {base}: insufficient valid data after NaN filtering.')
                continue

            X = x_proxy[mask].reshape(-1, 1)
            y = y_temp[mask]

            model     = LinearRegression().fit(X, y)
            slope     = float(model.coef_[0])
            intercept = float(model.intercept_)

            stats = _reg_stats(X.flatten(), y, f'{base} Final')
            self.final_equations[base] = {
                'slope':     slope,
                'intercept': intercept,
                'stats':     stats,
            }

            print(f'  {base} Final Model: Temp = {slope:.4f}*{base} + {intercept:.4f}')
            print(f'    R²={stats["r2"]:.3f}, RMSE={stats["rmse"]:.3f}, n={stats["n"]}\n')

    # ------------------------------------------------------------------
    # BLOCK BOOTSTRAP
    # ------------------------------------------------------------------

    def block_bootstrap_calibration(self, proxy='Mg/Sr', block_len=7, n_boot=1000):
        print('=' * 60)
        print(f'BLOCK BOOTSTRAP (proxy={proxy}, block_len={block_len}, n_boot={n_boot})')
        print('=' * 60)

        if proxy not in self.composite_data:
            print(f'❌ Composite data for {proxy} not found.\n')
            return

        comp         = self.composite_data[proxy]
        common_dates = comp.index.intersection(self.temp_full.index)
        x = comp.loc[common_dates, 'mean'].values
        y = self.temp_full.loc[common_dates].values
        n = len(x)
        if n < 20:
            print('❌ Too few days for meaningful bootstrap.\n')
            return

        slopes, intercepts, rmses = [], [], []
        starts = np.arange(0, n - block_len + 1)

        for _ in range(n_boot):
            idx = []
            while len(idx) < n:
                s = np.random.choice(starts)
                idx.extend(range(s, s + block_len))
            idx = np.array(idx[:n])

            xb  = x[idx]
            yb  = y[idx]
            msk = np.isfinite(xb) & np.isfinite(yb)
            if msk.sum() < 10:
                continue

            Xb   = xb[msk].reshape(-1, 1)
            yb   = yb[msk]
            model = LinearRegression().fit(Xb, yb)
            yhat  = model.predict(Xb)
            rmse  = np.sqrt(mean_squared_error(yb, yhat))

            slopes.append(float(model.coef_[0]))
            intercepts.append(float(model.intercept_))
            rmses.append(rmse)

        slopes     = np.array(slopes)
        intercepts = np.array(intercepts)
        rmses      = np.array(rmses)

        out = {
            'slope_ci':     np.percentile(slopes,     [2.5, 50, 97.5]),
            'intercept_ci': np.percentile(intercepts, [2.5, 50, 97.5]),
            'rmse_ci':      np.percentile(rmses,      [2.5, 50, 97.5]),
            'n_boot':       len(slopes),
        }
        self.bootstrap_results[proxy] = out

        print(f'  Slope 95% CI: {out["slope_ci"][0]:.4f} – {out["slope_ci"][2]:.4f}')
        print(f'  RMSE 95% CI:  {out["rmse_ci"][0]:.3f} – {out["rmse_ci"][2]:.3f}')
        print(f'  Effective bootstraps: {out["n_boot"]}\n')
        return out

    # ------------------------------------------------------------------
    # STEP 11: MULTI-ELEMENT SCREENING TABLE
    # ------------------------------------------------------------------

    def screen_all_elements_per_branch(self):
        """
        For every branch, align via Mg/Sr DTW, then regress each element
        ratio against temperature and print a ranked results table.
        """
        print('\n' + '=' * 70)
        print('STEP 11: SCREENING ALL ELEMENTS PER BRANCH (Mg/Sr Aligned)')
        print('=' * 70)

        if self.synthetic_master is None:
            print('❌ Master missing.')
            return

        master_vals  = self.synthetic_master['MgSr_Target'].values
        master_dates = self.synthetic_master['Date'].values
        window = int(self.config.get('dtw', {}).get('window', 10) or 10)
        psi    = self.config['dtw']['psi']

        for i in range(1, 8):
            df_raw, name = self.get_raw_branch_data(i)
            if df_raw is None:
                continue

            print(f'\n--- {name} ---')
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

                aligned_temps = self.temp_full.reindex(
                    master_dates[m_idx], method='nearest'
                ).values

                ratio_cols     = [c for c in df.columns if '/' in c and 'mmol' in c]
                branch_results = []
                df_subset      = df.loc[valid_mask].iloc[q_idx]

                for col in ratio_cols:
                    base     = col.split('(')[0].strip()
                    clean_el = ''.join([c for c in base if not c.isdigit()])
                    if '.' in clean_el:
                        clean_el = clean_el.split('.')[0]

                    vals = df_subset[col].values
                    mask = np.isfinite(vals) & np.isfinite(aligned_temps)

                    if mask.sum() < 20:
                        continue
                    if np.std(vals[mask]) == 0 or np.std(aligned_temps[mask]) == 0:
                        continue

                    slope, intercept, r_val, p_val, _ = linregress(
                        vals[mask], aligned_temps[mask]
                    )
                    branch_results.append({
                        'Element': clean_el,
                        'R2':      r_val ** 2,
                        'Slope':   slope,
                        'n':       mask.sum(),
                    })

                if branch_results:
                    res_df = pd.DataFrame(branch_results)
                    res_df = (
                        res_df
                        .groupby('Element')
                        .agg({'R2': 'mean', 'Slope': 'mean', 'n': 'mean'})
                        .reset_index()
                        .sort_values('R2', ascending=False)
                    )
                    print(res_df.to_string(index=False, float_format='%.3f'))
                else:
                    print('No valid regressions found.')

            except Exception as e:
                print(f'Skipped {name} due to alignment error: {e}')

    # ------------------------------------------------------------------
    # STEP 7 / 8 / 9: PLOT DELEGATION WRAPPERS
    # ------------------------------------------------------------------

    def plot_results(self):
        from rhodopipeline.plotting import plot_results
        plot_results(self)

    def plot_individual_branch_reconstructions(self):
        from rhodopipeline.plotting import plot_individual_branch_reconstructions
        plot_individual_branch_reconstructions(self)

    def plot_element_correlations_heatmap(self):
        from rhodopipeline.plotting import plot_element_correlations_heatmap
        plot_element_correlations_heatmap(self)
