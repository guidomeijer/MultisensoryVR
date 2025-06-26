import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# GPFA-related imports. You may need to install these:
# pip install elephant neo quantities
try:
    from elephant.gpfa import GPFA
    import neo
    import quantities as pq
    print("Successfully imported elephant, neo, and quantities.")
except ImportError:
    print("Please install elephant, neo, and quantities: pip install elephant neo quantities")
    class GPFA: pass
    class neo: pass
    class pq: pass

def simulate_raw_spike_data(n_neurons, n_trials, trial_duration_s, base_rate, event_trial, influence_strength=0.5):
    """
    Generates raw spike data: spike times, neuron IDs, and trial start times.
    """
    print("Simulating raw spike data...")
    total_duration_s = n_trials * trial_duration_s
    trial_start_times = np.arange(n_trials) * trial_duration_s

    # --- Region A: Generate base activity ---
    all_spike_times_A, all_spike_ids_A = [], []
    for i in range(n_neurons):
        n_spikes = np.random.poisson(base_rate * total_duration_s)
        spike_times = np.sort(np.random.uniform(0, total_duration_s, n_spikes))
        all_spike_times_A.extend(spike_times)
        all_spike_ids_A.extend([i] * len(spike_times))
    sort_idx = np.argsort(all_spike_times_A)
    spike_times_A = np.array(all_spike_times_A)[sort_idx]
    spike_ids_A = np.array(all_spike_ids_A)[sort_idx]

    # --- Region B: Generate activity with influence from A ---
    all_spike_times_B, all_spike_ids_B = [], []
    bin_size = 0.02
    bins = np.arange(0, total_duration_s + bin_size, bin_size)
    binned_A_rate, _ = np.histogram(spike_times_A, bins=bins)
    binned_A_rate = binned_A_rate / bin_size

    for i in range(n_neurons):
        # --- THIS IS THE FIX ---
        # Initialize rate_B as a float array to prevent type errors during addition.
        rate_B = np.full(len(bins)-1, base_rate, dtype=float)
        
        event_time = trial_start_times[event_trial]
        event_bin = int(event_time / bin_size)
        
        influence = influence_strength * binned_A_rate
        rate_B[event_bin:] += influence[event_bin:]
        
        integrated_rate = np.cumsum(rate_B * bin_size)
        max_rate = np.max(integrated_rate)
        if max_rate > 0:
            n_spikes_B = np.random.poisson(max_rate)
            spike_times_unmapped = np.random.uniform(0, max_rate, n_spikes_B)
            spike_times = np.interp(spike_times_unmapped, integrated_rate, bins[:-1])
            all_spike_times_B.extend(spike_times)
            all_spike_ids_B.extend([i] * len(spike_times))

    sort_idx = np.argsort(all_spike_times_B)
    spike_times_B = np.array(all_spike_times_B)[sort_idx]
    spike_ids_B = np.array(all_spike_ids_B)[sort_idx]
    
    print("Simulation complete.")
    return (spike_times_A, spike_ids_A), (spike_times_B, spike_ids_B), trial_start_times

def create_spiketrains_for_trial(spike_times, spike_ids, selected_neuron_ids, trial_start_time, trial_duration_s):
    """
    Creates a list of neo.SpikeTrain objects for a single trial from raw spike data,
    using only the neurons specified in selected_neuron_ids.
    
    Args:
        spike_times (np.ndarray): Array of all spike times.
        spike_ids (np.ndarray): Array of neuron IDs for each spike.
        selected_neuron_ids (list or np.ndarray): A list of neuron IDs to include.
        trial_start_time (float): The start time of the trial in seconds.
        trial_duration_s (float): The duration of the trial in seconds.

    Returns:
        list: A list of neo.SpikeTrain objects.
    """
    trial_spiketrains = []
    trial_end_time = trial_start_time + trial_duration_s
    
    # Iterate only through the selected neuron IDs
    for neuron_id in selected_neuron_ids:
        # 1. Filter all spikes to get just those from the current neuron
        neuron_mask = (spike_ids == neuron_id)
        neuron_spike_times = spike_times[neuron_mask]
        
        # 2. Filter the neuron's spikes to get just those in the current trial
        trial_mask = (neuron_spike_times >= trial_start_time) & (neuron_spike_times < trial_end_time)
        spikes_in_trial = neuron_spike_times[trial_mask]
        
        # 3. Make spike times relative to the start of the trial
        relative_spike_times = spikes_in_trial - trial_start_time
        
        # 4. Create the neo.SpikeTrain object with correct units and duration
        st = neo.SpikeTrain(relative_spike_times * pq.s, t_stop=trial_duration_s * pq.s)
        trial_spiketrains.append(st)
        
    return trial_spiketrains

def perform_gpfa(spiketrains, bin_size_s, n_components=3):
    """
    Performs GPFA directly on a list of SpikeTrain objects.
    """
    # Robustness check: Ensure there are enough active neurons to fit the model.
    active_neurons = [st for st in spiketrains if len(st) > 0]
    if len(active_neurons) < n_components:
        print(f"Skipping GPFA: Only {len(active_neurons)} active neurons found (need at least {n_components}).")
        # Determine number of bins from one of the spiketrains
        n_bins = int(spiketrains[0].t_stop / (bin_size_s * pq.s))
        return np.full((n_bins, n_components), np.nan)

    gpfa = GPFA(bin_size=bin_size_s * pq.s, x_dim=n_components)
    
    try:
        latent_variables = gpfa.fit_transform(spiketrains)
    except Exception as e:
        print(f"GPFA fitting failed: {e}")
        n_bins = int(spiketrains[0].t_stop / (bin_size_s * pq.s))
        return np.full((n_bins, n_components), np.nan)

    # Output is (n_components x n_bins), transpose to (n_bins x n_components)
    return latent_variables.T

def analyze_multivariate_granger_causality(data, max_lag, n_components):
    """Performs multivariate Granger causality test using a VAR model."""
    if data is None or data.shape[0] < max_lag + 5:
        return None
    try:
        model = VAR(data)
        results = model.fit(maxlags=max_lag, verbose=False)
        p_A_to_B = results.test_causality(np.arange(n_components, 2*n_components), np.arange(n_components), kind='f').pvalue
        p_B_to_A = results.test_causality(np.arange(n_components), np.arange(n_components, 2*n_components), kind='f').pvalue
        return {'A->B_pvalue': p_A_to_B, 'B->A_pvalue': p_B_to_A}
    except Exception as e:
        print(f"Granger causality failed: {e}")
        return None

# ==============================================================================
# --- Main Execution ---
# ==============================================================================
if __name__ == '__main__':
    # 1. Define Parameters
    N_NEURONS = 40
    N_TRIALS = 50
    TRIAL_DURATION_S = 2.0
    BASE_RATE_HZ = 10
    EVENT_TRIAL = N_TRIALS // 2
    INFLUENCE = 0.8
    BIN_SIZE_S = 0.02  # 20 ms bin size for GPFA
    N_COMPONENTS = 3
    MAX_LAG_GC = 5

    # 2. Simulate Data in Raw Format
    (st_A, sid_A), (st_B, sid_B), trial_starts = simulate_raw_spike_data(
        N_NEURONS, N_TRIALS, TRIAL_DURATION_S, BASE_RATE_HZ, EVENT_TRIAL, INFLUENCE)

    # Define which neurons to use from each region.
    # For example, use all neurons from Region A, but only the first half from Region B.
    neurons_to_use_A = np.arange(N_NEURONS)
    neurons_to_use_B = np.arange(N_NEURONS // 2)

    # 3. Loop through trials, create SpikeTrains, and run GPFA
    latents_a_before, latents_b_before = [], []
    latents_a_after, latents_b_after = [], []

    print("\n--- Processing Trials ---")
    for i in range(N_TRIALS):
        print(f"Trial {i+1}/{N_TRIALS}...")
        # Create SpikeTrain lists for this trial directly from raw data, using the selection
        spiketrains_a = create_spiketrains_for_trial(st_A, sid_A, neurons_to_use_A, trial_starts[i], TRIAL_DURATION_S)
        spiketrains_b = create_spiketrains_for_trial(st_B, sid_B, neurons_to_use_B, trial_starts[i], TRIAL_DURATION_S)
        
        # Run GPFA on the SpikeTrain lists
        lat_a = perform_gpfa(spiketrains_a, BIN_SIZE_S, N_COMPONENTS)
        lat_b = perform_gpfa(spiketrains_b, BIN_SIZE_S, N_COMPONENTS)
        
        # Store results if successful
        if not np.isnan(lat_a).any() and not np.isnan(lat_b).any():
            if i < EVENT_TRIAL:
                latents_a_before.append(lat_a)
                latents_b_before.append(lat_b)
            else:
                latents_a_after.append(lat_a)
                latents_b_after.append(lat_b)

    # 4. Aggregate latents and run Granger Causality
    if latents_a_before and latents_a_after:
        agg_latents_before = np.hstack([np.vstack(latents_a_before), np.vstack(latents_b_before)])
        agg_latents_after = np.hstack([np.vstack(latents_a_after), np.vstack(latents_b_after)])

        print("\n--- Analyzing Aggregated Granger Causality (Before Event) ---")
        gc_results_before = analyze_multivariate_granger_causality(agg_latents_before, MAX_LAG_GC, N_COMPONENTS)
        
        print("\n--- Analyzing Aggregated Granger Causality (After Event) ---")
        gc_results_after = analyze_multivariate_granger_causality(agg_latents_after, MAX_LAG_GC, N_COMPONENTS)

        # 5. Print Summary
        print("\n" + "="*30 + "\n      RESULTS SUMMARY\n" + "="*30)
        if gc_results_before:
            print(f"Before Event ({len(latents_a_before)} trials):")
            print(f"  A -> B p-value: {gc_results_before['A->B_pvalue']:.4f}")
            print(f"  B -> A p-value: {gc_results_before['B->A_pvalue']:.4f}")
        if gc_results_after:
            print(f"After Event ({len(latents_a_after)} trials):")
            print(f"  A -> B p-value: {gc_results_after['A->B_pvalue']:.4f}  <-- Expect low p-value")
            print(f"  B -> A p-value: {gc_results_after['B->A_pvalue']:.4f}")
        print("="*30)
    else:
        print("\nNot enough successful trials to perform final analysis.")

