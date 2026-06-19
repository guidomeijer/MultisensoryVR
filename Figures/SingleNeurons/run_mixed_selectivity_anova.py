# %%
# -*- coding: utf-8 -*-
"""
Conjunctive Mixed Selectivity Analysis using Two-Way ANOVA

For each neuron, builds a long-format "encounter table" where each row
is a single object encounter (Objects 1 & 2 only), with:
  - firing_rate : spike count in [0, T_EPOCH] post-entry divided by T_EPOCH (Hz)
  - object      : categorical ('obj1' vs 'obj2')
  - goal        : categorical ('goal' vs 'no-goal') — was this obj the rewarded one
                  on this trial?

A balanced two-way ANOVA is then run on firing_rate ~ C(object) + C(goal) +
C(object):C(goal), yielding p-values for:
  p_object      : main effect of object identity
  p_goal        : main effect of reward context
  p_interaction : interaction (conjunctive / mixed selectivity)

Results are saved to Data/mixed_selectivity_anova.csv.

Created by Guido Meijer  (2026)
"""

import numpy as np
from os.path import join
import pandas as pd
from joblib import Parallel, delayed
import statsmodels.formula.api as smf
from msvr_functions import paths, load_neural_data, load_subjects, load_objects

# ─────────────────────────── Settings ────────────────────────────────────────
T_START = 0.5
T_END = 1.0
OVERWRITE = True
N_CORES   = -6    # joblib: use all but 4 CPUs  (-1 = all cores)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────── Helper: spike count → firing rate for one encounter ──────────
def spikes_in_window(spike_times, t_start, t_stop):
    """Return spike count in (t_start, t_stop]."""
    return np.sum((spike_times > t_start) & (spike_times <= t_stop))


# ─────────────── Per-neuron ANOVA (called in parallel) ───────────────────────
def run_anova(neuron_id, encounter_df, spike_times):
    """
    Parameters
    ----------
    neuron_id    : int
    encounter_df : DataFrame with columns [times, object, goal]
                   restricted to rewarded objects (obj 1 & 2)
    spike_times  : 1-D array of spike times for this neuron

    Returns
    -------
    tuple : (p_object, p_goal, p_interaction,
             eta2_object, eta2_goal, eta2_interaction,
             n_trials)
    """
    nan_result = (np.nan,) * 7

    # ── Build the per-encounter firing rate vector ────────────────────────────
    rates = np.array([
        spikes_in_window(spike_times, t + T_START, t + T_END) / (T_END - T_START)
        for t in encounter_df['times'].values
    ])

    df = encounter_df[['object', 'goal']].copy()
    df['firing_rate'] = rates

    # ── Require at least 2 observations per cell (2×2 design = 4 cells) ──────
    counts = df.groupby(['object', 'goal']).size()
    if counts.min() < 2:
        return nan_result

    n_trials = len(df)

    # ── Two-way ANOVA ─────────────────────────────────────────────────────────
    try:
        from statsmodels.stats.anova import anova_lm
        model       = smf.ols('firing_rate ~ C(object) + C(goal) + C(object):C(goal)',
                              data=df).fit()
        anova_table = anova_lm(model, typ=2)   # Type II SS

        ss_object      = anova_table.loc['C(object)',          'sum_sq']
        ss_goal        = anova_table.loc['C(goal)',            'sum_sq']
        ss_interaction = anova_table.loc['C(object):C(goal)',  'sum_sq']
        ss_residual    = anova_table.loc['Residual',           'sum_sq']
        ss_total       = ss_object + ss_goal + ss_interaction + ss_residual

        p_object      = anova_table.loc['C(object)',          'PR(>F)']
        p_goal        = anova_table.loc['C(goal)',            'PR(>F)']
        p_interaction = anova_table.loc['C(object):C(goal)',  'PR(>F)']

        # Partial η²  =  SS_effect / (SS_effect + SS_residual)
        eta2_object      = ss_object      / (ss_object      + ss_residual)
        eta2_goal        = ss_goal        / (ss_goal        + ss_residual)
        eta2_interaction = ss_interaction / (ss_interaction + ss_residual)

    except Exception:
        return nan_result

    return (p_object, p_goal, p_interaction,
            eta2_object, eta2_goal, eta2_interaction,
            n_trials)


# ─────────────── Initialise ───────────────────────────────────────────────────
path_dict = paths()
subjects  = load_subjects()
rec       = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

# Optionally skip already-processed recordings
if OVERWRITE:
    anova_df = pd.DataFrame()
else:
    out_file = join(path_dict['save_path'], 'mixed_selectivity_anova.csv')
    try:
        anova_df = pd.read_csv(out_file)
        anova_df[['subject', 'date', 'probe']] = (
            anova_df[['subject', 'date', 'probe']].astype(str))
        merged = rec.merge(anova_df, on=['subject', 'date', 'probe'],
                           how='left', indicator=True)
        rec = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    except FileNotFoundError:
        anova_df = pd.DataFrame()


# ─────────────── Main loop over sessions ─────────────────────────────────────
# %%
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\n[{i+1}/{rec.shape[0]}]  {subject}  {date}  {probe}')

    # ── Load neural data ──────────────────────────────────────────────────────
    session_path = join(path_dict['local_data_path'], 'Subjects', subject, date)
    spikes, clusters, channels = load_neural_data(
        session_path, probe, histology=True, only_good=True, min_fr=0.1)

    # ── Load object-encounter data ────────────────────────────────────────────
    all_obj_df = load_objects(subject, date)

    # ── Keep only the two rewarded objects (1 & 2) ───────────────────────────
    #    'goal' column already codes whether that object was the rewarded one
    #    on that particular trial.
    rew_obj_df = all_obj_df[all_obj_df['object'].isin([1, 2])].copy()
    rew_obj_df['object'] = rew_obj_df['object'].map({1: 'obj1', 2: 'obj2'})
    rew_obj_df['goal']   = rew_obj_df['goal'].map({1: 'goal', 0: 'no_goal'})

    # ── Run ANOVA in parallel across neurons ──────────────────────────────────
    print(f'  Running two-way ANOVA on {clusters["cluster_id"].shape[0]} neurons …')
    results = Parallel(n_jobs=N_CORES)(
        delayed(run_anova)(
            nid,
            rew_obj_df[['times', 'object', 'goal']].reset_index(drop=True),
            spikes['times'][spikes['clusters'] == nid]
        )
        for nid in clusters['cluster_id']
    )

    # ── Unpack results ────────────────────────────────────────────────────────
    p_obj   = np.array([r[0] for r in results])
    p_goal  = np.array([r[1] for r in results])
    p_int   = np.array([r[2] for r in results])
    e2_obj  = np.array([r[3] for r in results])
    e2_goal = np.array([r[4] for r in results])
    e2_int  = np.array([r[5] for r in results])
    n_enc   = np.array([r[6] for r in results])

    # ── Append to output dataframe ────────────────────────────────────────────
    session_df = pd.DataFrame({
        'subject':       subject,
        'date':          date,
        'probe':         probe,
        'neuron_id':     clusters['cluster_id'],
        'region':        clusters['region'],
        'allen_acronym': clusters['acronym'],
        'x':             clusters['x'],
        'y':             clusters['y'],
        'z':             clusters['z'],
        # p-values (two-way ANOVA, Type II SS)
        'p_object':      p_obj,    # main effect: object identity (obj1 vs obj2)
        'p_goal':        p_goal,   # main effect: reward context (goal vs no-goal)
        'p_interaction': p_int,    # interaction: conjunctive / mixed selectivity
        # Effect sizes (partial η²)
        'eta2_object':      e2_obj,
        'eta2_goal':        e2_goal,
        'eta2_interaction': e2_int,
        # Diagnostics
        'n_encounters':  n_enc,    # total object encounters entering the ANOVA
        't_win_start_s':     T_START,  # epoch window used
        't_win_end_s':     T_END,  # epoch window used
    })

    anova_df = pd.concat([anova_df, session_df], ignore_index=True)

    # ── Save after every session ──────────────────────────────────────────────
    anova_df.to_csv(join(path_dict['save_path'], 'mixed_selectivity_anova.csv'),
                    index=False)
    print(f'  Saved → mixed_selectivity_anova.csv')

print('\nDone.')
