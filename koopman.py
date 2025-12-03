# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:53:45 2025

By Gemini 3 Pro
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d


class NeuralKoopmanPipeline:
    def __init__(self, dt=0.01, sigma=0.05, n_delays=10, rank=10):
        """
        Args:
            dt: Bin size in seconds
            sigma: Width of Gaussian smoothing kernel in seconds
            n_delays: Number of history steps to stack (Time-Delay Embedding)
            rank: Rank for Reduced Rank Regression (if using reduction)
        """
        self.dt = dt
        self.sigma = sigma
        self.n_delays = n_delays
        self.rank = rank
        self.models = {}
        
    def _process_single_trial(self, times, ids, area_map):
        """
        Internal helper: Converts a SINGLE trial of spikes into smooth trajectories.
        """
        if len(times) == 0:
            return None, None

        # 1. Determine time bounds (relative to trial start 0)
        t_max = np.max(times)
        n_bins = int(np.ceil(t_max / self.dt))
        
        # Ensure we have enough bins for the delay embedding
        if n_bins <= self.n_delays:
            return None, None

        # 2. Binning
        neurons_A = area_map['A']
        neurons_B = area_map['B']
        
        id_map_A = {nid: i for i, nid in enumerate(neurons_A)}
        id_map_B = {nid: i for i, nid in enumerate(neurons_B)}
        
        binned_A = np.zeros((n_bins, len(neurons_A)))
        binned_B = np.zeros((n_bins, len(neurons_B)))
        
        for t, nid in zip(times, ids):
            bin_idx = int(t / self.dt)
            if bin_idx < n_bins:
                if nid in id_map_A:
                    binned_A[bin_idx, id_map_A[nid]] += 1
                elif nid in id_map_B:
                    binned_B[bin_idx, id_map_B[nid]] += 1
                    
        # 3. Smoothing
        sigma_bins = self.sigma / self.dt
        X_A = gaussian_filter1d(binned_A, sigma_bins, axis=0)
        X_B = gaussian_filter1d(binned_B, sigma_bins, axis=0)
        
        # Normalize (Standard Score) - vital for Ridge Regression
        # Note: We return raw matrices here; normalization usually happens 
        # globally across all trials to preserve relative amplitude, 
        # but trial-wise normalization is acceptable if firing rates vary wildly.
        # Here we do trial-wise for simplicity.
        X_A = (X_A - X_A.mean(axis=0)) / (X_A.std(axis=0) + 1e-6)
        X_B = (X_B - X_B.mean(axis=0)) / (X_B.std(axis=0) + 1e-6)
        
        return X_A, X_B

    def preprocess_trials(self, trials_data, area_map):
        """
        Processes a list of trials.
        
        Args:
            trials_data: List of tuples [(times_1, ids_1), (times_2, ids_2), ...]
            area_map: Dict mapping Area IDs
            
        Returns:
            trials_X_A: List of trajectory matrices for Area A
            trials_X_B: List of trajectory matrices for Area B
        """
        trials_X_A = []
        trials_X_B = []
        
        print(f"Processing {len(trials_data)} trials...")
        
        for i, (t, nid) in enumerate(trials_data):
            X_A, X_B = self._process_single_trial(t, nid, area_map)
            if X_A is not None:
                trials_X_A.append(X_A)
                trials_X_B.append(X_B)
            else:
                print(f"Warning: Trial {i} was too short or empty. Skipped.")
                
        return trials_X_A, trials_X_B

    def get_stacked_hankel(self, trials_X):
        """
        Constructs Hankel matrices for each trial and vertically stacks them.
        This prevents 'ghost' dynamics between trials.
        
        Args:
            trials_X: List of trajectory matrices [X_trial1, X_trial2, ...]
            
        Returns:
            H_stacked: The input history matrix (N_total_samples x (Features*Delays))
            Y_stacked: The target future matrix (N_total_samples x Features)
        """
        H_list = []
        Y_list = []
        
        for X in trials_X:
            n_samples, n_features = X.shape
            valid_samples = n_samples - self.n_delays
            
            if valid_samples <= 0:
                continue

            # Create Hankel for this specific trial
            H_trial = np.zeros((valid_samples, n_features * self.n_delays))
            for i in range(self.n_delays):
                H_trial[:, i*n_features : (i+1)*n_features] = X[i : i+valid_samples]
            
            # Target is the state at t (current step of the prediction window)
            # Typically Koopman predicts t+1 given t. 
            # In delay coordinates: Pred Y[t] given H[t-1...t-d]
            Y_trial = X[self.n_delays:]
            
            H_list.append(H_trial)
            Y_list.append(Y_trial)
            
        if not H_list:
            raise ValueError("No valid data after Hankel processing.")
            
        return np.vstack(H_list), np.vstack(Y_list)

    def fit_koopman_operators(self, trials_X_A, trials_X_B):
        """
        Fits operators on the stacked data from all trials.
        """
        # 1. Get Stacked Matrices
        H_A, Y_A = self.get_stacked_hankel(trials_X_A)
        H_B, Y_B = self.get_stacked_hankel(trials_X_B)
        
        # Joint State
        H_AB = np.hstack([H_A, H_B])
        
        # 2. Fit Operators using Ridge Regression
        alpha = 1.0 
        print(f"Fitting Koopman on {H_A.shape[0]} pooled time points...")
        
        self.models['A'] = Ridge(alpha=alpha, fit_intercept=False).fit(H_A, Y_A)
        self.models['B'] = Ridge(alpha=alpha, fit_intercept=False).fit(H_B, Y_B)
        
        self.models['Joint_to_B'] = Ridge(alpha=alpha, fit_intercept=False).fit(H_AB, Y_B)
        self.models['Joint_to_A'] = Ridge(alpha=alpha, fit_intercept=False).fit(H_AB, Y_A)

    def analyze_causality(self, trials_X_A, trials_X_B):
        """
        Computes Koopman Granger Causality across all trials.
        """
        # 1. Prepare Stacked Test Data
        H_A, Y_A = self.get_stacked_hankel(trials_X_A)
        H_B, Y_B = self.get_stacked_hankel(trials_X_B)
        H_AB = np.hstack([H_A, H_B])
        
        # 2. Evaluate Marginal Predictions
        pred_B_marginal = self.models['B'].predict(H_B)
        mse_B_marginal = mean_squared_error(Y_B, pred_B_marginal)
        
        pred_A_marginal = self.models['A'].predict(H_A)
        mse_A_marginal = mean_squared_error(Y_A, pred_A_marginal)
        
        # 3. Evaluate Joint Predictions
        pred_B_joint = self.models['Joint_to_B'].predict(H_AB)
        mse_B_joint = mean_squared_error(Y_B, pred_B_joint)
        
        pred_A_joint = self.models['Joint_to_A'].predict(H_AB)
        mse_A_joint = mean_squared_error(Y_A, pred_A_joint)
        
        # 4. Compute Causality Metric
        causality_A_to_B = max(0, (mse_B_marginal - mse_B_joint) / mse_B_marginal)
        causality_B_to_A = max(0, (mse_A_marginal - mse_A_joint) / mse_A_marginal)
        
        return causality_A_to_B, causality_B_to_A

    def plot_eigenvalues(self):
        """
        Plots eigenvalues/singular values.
        """
        if not self.models:
            print("Models not fitted yet.")
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        
        K_A = self.models['A'].coef_
        K_B = self.models['B'].coef_

        # SVD proxy for mode strength
        _, s_A, _ = np.linalg.svd(K_A)
        _, s_B, _ = np.linalg.svd(K_B)
        
        ax.plot(s_A, label='Singular Values (Area A)', marker='o')
        ax.plot(s_B, label='Singular Values (Area B)', marker='x')
        
        ax.set_title("Dynamical Mode Strength (Multi-Trial)")
        ax.set_xlabel("Mode Index")
        ax.set_ylabel("Magnitude")
        ax.legend()
        plt.show()