# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 2026

By Guido Meijer
"""

import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from joblib import Parallel, delayed
from msvr_functions import paths, load_subjects

# Settings
BIN_MIN = 1100   # mm (start of approach to second object)
BIN_MAX = 1350  # mm (just before second object)
N_CORES = -4
OVERWRITE = True

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
far_subjects = subjects.loc[subjects['Far'] == 1, 'SubjectID'].astype(str).values

# Load in processed residual data
with open(path_dict['google_drive_data_path'] / 'residuals_position_20mms.pickle', 'rb') as handle:
    residuals_dict = pickle.load(handle)


def fit_two_way_anova(y, spatial_bins, contexts):
    """
    Fits two-way ANOVA on residual firing rates:
    fr ~ C(spatial_bin) + C(context) + C(spatial_bin):C(context)
    """
    if len(y) == 0 or np.all(np.isnan(y)) or np.nanstd(y) == 0:
        return {
            'p_spatial_bin': np.nan, 'f_spatial_bin': np.nan, 'ss_spatial_bin': np.nan, 'eta_sq_spatial_bin': np.nan,
            'p_context': np.nan, 'f_context': np.nan, 'ss_context': np.nan, 'eta_sq_context': np.nan,
            'p_interaction': np.nan, 'f_interaction': np.nan, 'ss_interaction': np.nan, 'eta_sq_interaction': np.nan,
            'ss_residual': np.nan
        }

    df_neuron = pd.DataFrame({
        'fr': y,
        'spatial_bin': spatial_bins.astype(str),
        'context': contexts.astype(str)
    }).dropna()

    if df_neuron['spatial_bin'].nunique() < 2 or df_neuron['context'].nunique() < 2:
        return {
            'p_spatial_bin': np.nan, 'f_spatial_bin': np.nan, 'ss_spatial_bin': np.nan, 'eta_sq_spatial_bin': np.nan,
            'p_context': np.nan, 'f_context': np.nan, 'ss_context': np.nan, 'eta_sq_context': np.nan,
            'p_interaction': np.nan, 'f_interaction': np.nan, 'ss_interaction': np.nan, 'eta_sq_interaction': np.nan,
            'ss_residual': np.nan
        }

    try:
        model = ols('fr ~ C(spatial_bin) + C(context) + C(spatial_bin):C(context)', data=df_neuron).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        ss_total = anova_table['sum_sq'].sum()
        ss_bin = anova_table.loc['C(spatial_bin)', 'sum_sq']
        ss_ctx = anova_table.loc['C(context)', 'sum_sq']
        ss_inter = anova_table.loc['C(spatial_bin):C(context)', 'sum_sq']
        ss_res = anova_table.loc['Residual', 'sum_sq']

        p_bin = anova_table.loc['C(spatial_bin)', 'PR(>F)']
        f_bin = anova_table.loc['C(spatial_bin)', 'F']
        eta_bin = ss_bin / ss_total if ss_total > 0 else np.nan

        p_ctx = anova_table.loc['C(context)', 'PR(>F)']
        f_ctx = anova_table.loc['C(context)', 'F']
        eta_ctx = ss_ctx / ss_total if ss_total > 0 else np.nan

        p_inter = anova_table.loc['C(spatial_bin):C(context)', 'PR(>F)']
        f_inter = anova_table.loc['C(spatial_bin):C(context)', 'F']
        eta_inter = ss_inter / ss_total if ss_total > 0 else np.nan

        return {
            'p_spatial_bin': p_bin,
            'f_spatial_bin': f_bin,
            'ss_spatial_bin': ss_bin,
            'eta_sq_spatial_bin': eta_bin,
            'p_context': p_ctx,
            'f_context': f_ctx,
            'ss_context': ss_ctx,
            'eta_sq_context': eta_ctx,
            'p_interaction': p_inter,
            'f_interaction': f_inter,
            'ss_interaction': ss_inter,
            'eta_sq_interaction': eta_inter,
            'ss_residual': ss_res
        }
    except Exception:
        return {
            'p_spatial_bin': np.nan, 'f_spatial_bin': np.nan, 'ss_spatial_bin': np.nan, 'eta_sq_spatial_bin': np.nan,
            'p_context': np.nan, 'f_context': np.nan, 'ss_context': np.nan, 'eta_sq_context': np.nan,
            'p_interaction': np.nan, 'f_interaction': np.nan, 'ss_interaction': np.nan, 'eta_sq_interaction': np.nan,
            'ss_residual': np.nan
        }


# %% Loop over recordings
anova_records = []
n_recordings = len(residuals_dict['residuals'])

for i in range(n_recordings):
    this_subject = str(residuals_dict['subject'][i])
    this_date = str(residuals_dict['date'][i])
    this_probe = str(residuals_dict['probe'][i])

    # Filter for FAR sessions only
    if this_subject not in far_subjects:
        continue

    print(f'Processing {this_subject} {this_date} {this_probe} [{i+1} of {n_recordings}]')

    spatial_bins = residuals_dict['position'][i]
    context = residuals_dict['context'][i]
    residuals = residuals_dict['residuals'][i]  # shape: (n_samples, n_neurons)
    neuron_ids = residuals_dict['neuron_id'][i]
    regions = residuals_dict['region'][i]
    acronyms = residuals_dict['acronym'][i] if 'acronym' in residuals_dict else [None] * len(neuron_ids)

    # Select non-overlapping spatial bins leading up to the second object (900 to 1300 mm)
    spatial_mask = (spatial_bins >= BIN_MIN) & (spatial_bins <= BIN_MAX) & (spatial_bins % 25 == 0)

    if not np.any(spatial_mask):
        continue

    sub_spatial_bins = spatial_bins[spatial_mask]
    sub_context = context[spatial_mask]
    sub_residuals = residuals[spatial_mask, :]
    n_neurons = sub_residuals.shape[1]

    # Parallel two-way ANOVA per neuron
    results = Parallel(n_jobs=N_CORES)(
        delayed(fit_two_way_anova)(
            sub_residuals[:, n_idx],
            sub_spatial_bins,
            sub_context
        ) for n_idx in range(n_neurons)
    )

    for n_idx, res in enumerate(results):
        record = {
            'subject': this_subject,
            'date': this_date,
            'probe': this_probe,
            'neuron_id': neuron_ids[n_idx],
            'region': regions[n_idx] if regions is not None else np.nan,
            'acronym': acronyms[n_idx] if acronyms is not None else np.nan,
            **res
        }
        anova_records.append(record)

# %% Save results to disk
anova_df = pd.DataFrame(anova_records)
save_file = path_dict['save_path'] / 'two_way_anova_residuals.csv'
anova_df.to_csv(save_file, index=False)
print(f'Done! Saved {len(anova_df)} neurons to {save_file}')
