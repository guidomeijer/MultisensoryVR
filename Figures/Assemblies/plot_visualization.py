# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 13/04/2026
"""
# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from msvr_functions import paths, figure_style
colors, dpi = figure_style()
path_dict = paths()

# 1. Generate Simulated, Correlated Assembly Data
np.random.seed(42)

# We simulate Z-scored data. Class 0 (Non-Rewarded) and Class 1 (Rewarded)
# have different means, but share a strong positive covariance (correlated noise).
mean_0 = [-1.5, -2.0]
mean_1 = [1.5, 2.0]

# High off-diagonal covariance: Assemblies A and B tend to fire together
# regardless of the trial outcome.
cov = [[0.7, 1.8],
       [1.8, 0.7]]

# Generate 200 trials for each condition
X_0 = np.random.multivariate_normal(mean_0, cov, 200)
X_1 = np.random.multivariate_normal(mean_1, cov, 200)
X = np.vstack((X_0, X_1))
y = np.hstack((np.zeros(200), np.ones(200)))

# 2. Fit the Regularized LDA
# Using 'eigen' and 'auto' shrinkage as recommended to prevent overfitting
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
lda.fit(X, y)

# The LDA coefficient vector (the axis of maximum separation)
lda_coef = lda.coef_[0]

# 3. Define a hypothetical Ripple Event
# During a ripple, firing rates burst. This pattern matches the Rewarded
# state but with much higher absolute magnitude.
ripple_state = [np.array([2.1, 2.8]), np.array([-2.4, 1.9]), np.array([-0.8, -1.1])]

# Calculate dot product
dot_prod = np.dot(ripple_state, lda_coef) / np.linalg.norm(lda_coef)

# %% Plotting
plot_colors = ['tab:orange', 'tab:purple', 'tab:blue']

fig, ax = plt.subplots(figsize=(3, 3), dpi=dpi)
ax.scatter(X_1[:, 0], X_1[:, 1], alpha=0.5, color=colors['goal'], label='Rewarded trials', edgecolors='none')
ax.scatter(X_0[:, 0], X_0[:, 1], alpha=0.5, color=colors['no-goal'], label='Non-Rewarded trials', edgecolors='none')

# Formatting
ax.axhline(0, color='grey', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5)
ax.set(xticks=[0], yticks=[0])
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel('Assembly A amplitude (Z-Scored)')
ax.set_ylabel('Assembly B amplitude (Z-Scored)')
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'viz_1.jpg', dpi=600)

# Plot the LDA Coefficient Vector from Origin
ax.quiver(0, 0, lda_coef[0], lda_coef[1], angles='xy', scale_units='xy', scale=1,
          color='black', width=0.006)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'viz_2.jpg', dpi=600)

# Plot the Ripple Vector from Origin
ax.scatter(ripple_state[0][0], ripple_state[0][1], color=plot_colors[0], s=20)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'viz_3.jpg', dpi=600)
ax.quiver(0, 0, ripple_state[0][0], ripple_state[0][1], angles='xy', scale_units='xy', scale=1,
          color=plot_colors[0], width=0.006)
ax.text(1.8, -3, f'Dot product: {dot_prod[0]:.1f}', color=plot_colors[0], fontweight='bold')
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'viz_4.jpg', dpi=600)

ax.scatter(ripple_state[1][0], ripple_state[1][1], color=plot_colors[1], s=20)
ax.quiver(0, 0, ripple_state[1][0], ripple_state[1][1], angles='xy', scale_units='xy', scale=1,
          color=plot_colors[1], width=0.006)
ax.text(1.8, -3.5, f'Dot product: {dot_prod[1]:.1f}', color=plot_colors[1], fontweight='bold')
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'viz_5.jpg', dpi=600)

ax.scatter(ripple_state[2][0], ripple_state[2][1], color=plot_colors[2], s=20)
ax.quiver(0, 0, ripple_state[2][0], ripple_state[2][1], angles='xy', scale_units='xy', scale=1,
          color=plot_colors[2], width=0.006)
ax.text(1.8, -4, f'Dot product: {dot_prod[2]:.1f}', color=plot_colors[2], fontweight='bold')
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'viz_6.jpg', dpi=600)

plt.show()

