import numpy as np
import pandas as pd
import pickle
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, load_objects, figure_style
colors, dpi = figure_style()

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()

# Settings
FIRST_OBJ = 450 + 100
NEAR_OBJ = 900 - 100
FAR_OBJ = 1350 - 100

# Load in processed data
with open(path_dict['google_drive_data_path'] / 'residuals_motor.pickle', 'rb') as handle:
    residuals_dict = pickle.load(handle)

# Loop over recordings
results_df = pd.DataFrame()
for i in range(len(residuals_dict['residuals'])):
    print(f'Recording {i} of {len(residuals_dict["residuals"])}')

    # Get subject
    subject = residuals_dict['subject'][i]
    date = residuals_dict['date'][i]
    is_far = subjects.loc[subjects['SubjectID'] == subject, 'Far'].values[0].astype(bool)
    if is_far:
        second_obj = FAR_OBJ
    else:
        second_obj = NEAR_OBJ

    # Get which context is the rewarded context for the first and second object
    obj_df = load_objects(subject, date)
    obj1_goal = obj_df.loc[(obj_df['object'] == 1) & (obj_df['goal'] == 1), 'sound'].values[0]
    obj2_goal = obj_df.loc[(obj_df['object'] == 2) & (obj_df['goal'] == 1), 'sound'].values[0]

    # Get position bins
    rel_pos_bins = np.unique(residuals_dict['position'][i]).astype(int)

    # Loop over neurons
    has_peak = np.full(residuals_dict['residuals'][i].shape[1], False)
    pos_peak = np.full(residuals_dict['residuals'][i].shape[1], False)
    p_values = np.full(residuals_dict['residuals'][i].shape[1], np.nan)
    for n in range(residuals_dict['residuals'][i].shape[1]):
        this_region = residuals_dict['region'][i][n]
        if this_region == 'root':
            continue

        # Get average activity per position
        this_df = pd.DataFrame({'position': residuals_dict['position'][i],
                                'activity': residuals_dict['residuals'][i][:, n],
                                'context': residuals_dict['context'][i]})
        this_df = this_df[this_df['context'] != 0]
        this_df = this_df[(this_df['position'] >= FIRST_OBJ) &
                          (this_df['position'] <= second_obj)]
        context_mean = this_df.groupby(['position', 'context'])['activity'].mean().reset_index()

        # Check if there is a significant peak in the rewarded context
        has_peak[n] = (context_mean[context_mean['context'] == obj2_goal]['activity'].max()
                       > 2 * context_mean[context_mean['context'] == obj2_goal]['activity'].std())

        # Check significant difference between rewarded and unrewarded
        peak_position = context_mean.loc[context_mean['activity'].idxmax(), 'position']
        _, p_values[n] = stats.ttest_ind(
            this_df[(this_df['position'] == peak_position) & (this_df['context'] == 1)]['activity'].values,
            this_df[(this_df['position'] == peak_position) & (this_df['context'] == 2)]['activity'].values)

        # Check if peak is higher for rewarded
        rew_peak = context_mean[(context_mean['position'] == peak_position) & (context_mean['context'] == obj2_goal)]['activity'].values[0]
        unrew_peak = context_mean[(context_mean['position'] == peak_position) & (context_mean['context'] == obj1_goal)]['activity'].values[0]
        pos_peak[n] = rew_peak > unrew_peak

        # If the peak is more than 3 stds plot neuron
        if has_peak[n] and p_values[n] < 0.05 and pos_peak[n]:
            
            # Plot this neuron
            fig, ax1 = plt.subplots(figsize=(2, 2))
            sns.lineplot(data=this_df, x='position', y='activity', hue='context', errorbar='se',
                         ax=ax1, err_kws={'lw': 0}, hue_order=[obj2_goal, obj1_goal],
                         palette=[colors['goal'], colors['no-goal']], legend=False)
            ax1.set(ylabel='Residual neural activity (spks/s)', xlabel='Position (mm)')
            sns.despine(trim=False)
            plt.tight_layout()
            plt.savefig(path_dict['fig_path'] / 'ExampleNeurons' / 'RewardField'
                        / f'{this_region}_{subject}_{date}_neuron{n}.jpg', dpi=600)
            plt.close()

    # Add to dataframe
    reward_field_neurons = has_peak & (p_values < 0.05) & pos_peak
    results_df = pd.concat((results_df, pd.DataFrame(data={
        'has_reward_field': reward_field_neurons, 'neuron_id': residuals_dict['neuron_id'][i],
        'subject': subject, 'date': date, 'region': residuals_dict['region'][i]
    })))

# %% Plot summary
results_df = results_df[results_df['region'] != 'root']
per_ses_df = results_df.groupby(['region', 'date']).sum(numeric_only=True)
per_ses_df['n_neurons'] = results_df.groupby(['region', 'date']).size()
per_ses_df['perc_reward_field'] = (per_ses_df['has_reward_field'] / per_ses_df['n_neurons']) * 100
per_ses_df = per_ses_df.reset_index()

f, ax1 = plt.subplots(figsize=(2, 2))
this_order = per_ses_df[['region', 'perc_reward_field']].groupby('region').mean().sort_values(
    'perc_reward_field', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_reward_field', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.set(ylabel='Significant neurons (%)', xlabel='',
        title='Neurons with reward field')
ax1.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(path_dict['fig_path'] / 'ExampleNeurons' / 'RewardField' / 'RewardField_summary.jpg', dpi=600)


        
        
