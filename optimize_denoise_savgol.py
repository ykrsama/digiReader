import numpy as np
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from scipy.stats import moyal
from sklearn.mixture import GaussianMixture
from scipy.optimize import curve_fit
import itertools
import plotly.graph_objects as go
from tools import plot


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
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=noisy_waveform, mode='lines', name='Noisy Waveform', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=x, y=clean_waveform, mode='lines', name='Clean Waveform'))
fig.add_trace(go.Scatter(x=x, y=optimized_smoothed_waveform, mode='lines',
                         name='Optimized Smoothed Waveform', line=dict(dash='dash')))
fig.add_hline(y=best_gmm_baseline, line=dict(color='goldenrod', dash='dashdot'),
              annotation=dict(text="GMM Baseline", xshift=-400, font=dict(color='goldenrod', size=16)))
fig.add_hline(y=baseline_landau, line=dict(color='brown', dash='dot'),
              annotation=dict(text="Landau Baseline", font=dict(color='brown', size=16)))

dy = np.max(noisy_waveform) - np.min(noisy_waveform)

plot.plot_style(
    fig,
    yaxis_range=[np.min(noisy_waveform) - 0.03 * dy , np.max(noisy_waveform) + 0.3 * dy],
    title='Waveform Analysis', xaxis_title='Time', yaxis_title='Amplitude',
    darkshine_label2=f'Window Length: {best_window_length}, Polyorder: {best_polyorder}<br>' +
                     f'GMM Component N: {best_n}'
)

fig.show()