# -*- coding: utf-8 -*-
"""
Illustration of how the LDA context axis in region_corr_lda.py works, using 2D mock data.

Two clouds of points (context A and context B, e.g. the activity of two neurons on single time
bins) are generated with a shared, correlated within-class covariance. LDA is fit with the same
settings as in region_corr_lda.py and the LDA axis is drawn through the clouds. Two example points
are projected onto the axis and their LDA score is calculated with the same function as in the
analysis: the distance along the unit-length discriminant axis, centered on the midpoint between
the class means (positive = context B).

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from msvr_functions import paths, figure_style

colors, dpi = figure_style()

# Settings (LDA settings match region_corr_lda.py)
SHRINKAGE = 'auto'
N_POINTS = 150  # points per context
SEED = 42
COLOR_A, COLOR_B = '#1f77b4', '#d62728'  # context A, context B
path_dict = paths()

# Generate mock data: shared correlated covariance, means offset along a diagonal direction
rng = np.random.default_rng(SEED)
cov = np.array([[1.0, 0.8], [0.8, 1.0]])
mean_A, mean_B = np.array([-0.6, 0.6]), np.array([0.6, -0.1])
X_A = rng.multivariate_normal(mean_A, cov, N_POINTS)
X_B = rng.multivariate_normal(mean_B, cov, N_POINTS)
X = np.vstack((X_A, X_B))
y = np.concatenate((np.ones(N_POINTS), 2 * np.ones(N_POINTS))).astype(int)

# Fit LDA, same as in the analysis (equal priors so the boundary is at the class mean midpoint)
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=SHRINKAGE, priors=[0.5, 0.5])
lda.fit(X, y)


def lda_projection(model, X):
    """
    Project X onto the unit-length LDA axis, centered on the midpoint between the class means.
    With equal priors the decision function is w.x - w.(mu_A + mu_B) / 2, so dividing it by |w|
    gives the distance along the discriminant axis (positive = context B).
    """
    return model.decision_function(X) / np.linalg.norm(model.coef_[0])


# LDA axis: unit vector through the midpoint between the class means
w_unit = lda.coef_[0] / np.linalg.norm(lda.coef_[0])
midpoint = lda.means_.mean(axis=0)

# Two example points (one closer to each context) and their scores
examples = np.array([[-1.2, 0.2], [1.3, 0.9]])
scores = lda_projection(lda, examples)
projected = midpoint + scores[:, None] * w_unit  # location of the projection in the 2D space
for i, s in enumerate(scores):
    print(f'Example point {i + 1}: LDA score = {s:.2f}')

# %% Plot
f, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)

# Clouds
ax.scatter(*X_A.T, s=6, color=COLOR_A, alpha=0.5, lw=0, label='Context A')
ax.scatter(*X_B.T, s=6, color=COLOR_B, alpha=0.5, lw=0, label='Context B')

# LDA axis through the midpoint
lim = 3.2
ax.plot(*(midpoint[:, None] + np.array([-lim, lim]) * w_unit[:, None]), color='k', lw=1,
        zorder=1)
ax.plot(*midpoint, marker='+', color='k', ms=5, zorder=3)  # score = 0

# Decision boundary (score = 0), perpendicular to the axis
w_perp = np.array([-w_unit[1], w_unit[0]])
ax.plot(*(midpoint[:, None] + np.array([-lim, lim]) * w_perp[:, None]), color='grey', lw=0.5,
        ls='--', zorder=0)

# Example points, projection lines and projected points
for ex, pr, score in zip(examples, projected, scores):
    ax.plot([ex[0], pr[0]], [ex[1], pr[1]], color='grey', lw=0.75, ls=':', zorder=2)
    ax.scatter(*ex, s=25, color='w', edgecolor='k', lw=1, zorder=4)
    ax.scatter(*pr, s=20, color='k', zorder=4)
    ax.annotate(f'score = {score:.2f}', xy=pr, xytext=(4, -9 if score > 0 else 6),
                textcoords='offset points', fontsize=6, ha='left' if score > 0 else 'right')

ax.set(xlabel='Neuron 1 activity (z)', ylabel='Neuron 2 activity (z)',
       xlim=[-lim, lim], ylim=[-lim, lim], aspect='equal',
       title='LDA axis and projection')
ax.legend(frameon=False, loc='upper left', fontsize=6, handletextpad=0.2)
sns.despine(trim=True)
plt.tight_layout()
# plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'lda_example.pdf')
plt.show()
