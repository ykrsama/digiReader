import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from scipy.stats import moyal
from sklearn.mixture import GaussianMixture
from scipy.optimize import curve_fit
import itertools

# Generate a sample waveform
np.random.seed(0)
x = np.linspace(0, 200, 200)
clean_waveform = 4000 * moyal.pdf(x, 40, 10)
noisy_waveform = clean_waveform + np.random.normal(0, 4.2, len(clean_waveform))

# Define parameter ranges
window_lengths = range(5, 50, 2)
poly_orders = range(2, 6)

# Optimize parameters
best_mse = float('inf')
best_params = None

for window_length, polyorder in itertools.product(window_lengths, poly_orders):
    # Apply Savitzky-Golay filter
    try:
        smoothed_waveform = savgol_filter(noisy_waveform, window_length, polyorder)
        mse = mean_squared_error(clean_waveform, smoothed_waveform)
        if mse < best_mse:
            best_mse = mse
            best_params = (window_length, polyorder)
    except ValueError:
        continue

# Apply the best parameters
best_window_length, best_polyorder = best_params
optimized_smoothed_waveform = savgol_filter(noisy_waveform, best_window_length, best_polyorder)

print(best_params)

# Estimate baseline using Gaussian Mixture
best_delta = float('inf')
best_n = None
best_gmm_baseline = None
gmm_ns = range(2, 10)

for n in list(gmm_ns):
    gmm = GaussianMixture(n_components=n)
    gmm.fit(optimized_smoothed_waveform.reshape(-1, 1))
    gmm_baseline = gmm.means_.min()
    if abs(gmm_baseline) < best_delta:
        best_delta = abs(gmm_baseline)
        best_gmm_baseline = gmm_baseline
        best_n = n

print(best_n)

def landau_func(x, loc, scale, amplitude, baseline):
    return amplitude * moyal.pdf(x, loc=loc, scale=scale) + baseline

def estimate_baseline_landau(signal):
    # Initial parameter guess
    max_index = np.argmax(signal)
    loc_guess = x[max_index]
    half_max = (signal[max_index] + signal.min()) / 2
    left_index = np.where(signal[:max_index] <= half_max)[0]
    right_index = np.where(signal[max_index:] <= half_max)[0] + max_index

    if len(left_index) > 0 and len(right_index) > 0:
        scale_guess = (x[right_index[0]] - x[left_index[-1]]) / 2
    else:
        scale_guess = 5
    amplitude_guess = signal[max_index]
    baseline_guess = signal.min()
    initial_guess = [loc_guess, scale_guess, amplitude_guess, baseline_guess]

    # Fit the noisy waveform
    popt, _ = curve_fit(landau_func, x, signal, p0=initial_guess)

    # Extract the estimated baseline
    return popt[-1]

baseline_landau = estimate_baseline_landau(optimized_smoothed_waveform)

# Plot the results
plt.plot(x, noisy_waveform, label="Noisy Waveform", linestyle=':')
plt.plot(x, clean_waveform, label="Clean Waveform")
plt.plot(x, optimized_smoothed_waveform, label="Optimized Smoothed Waveform", linestyle='--')
plt.axhline(y=baseline_landau, color='orange', linestyle='-.', label="Baseline Landau")
plt.axhline(y=best_gmm_baseline, color='r', linestyle='-.', label="Baseline GMM")
plt.legend()
plt.show()


