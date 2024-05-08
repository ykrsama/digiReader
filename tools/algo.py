import warnings
import numpy as np
from scipy.stats import moyal
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.mixture import GaussianMixture


def estimate_baseline_gmm(waveform, n=9):
    # Estimate baseline using Gaussian Mixture
    gmm = GaussianMixture(n_components=n)
    gmm.fit(waveform.reshape(-1, 1))
    return gmm.means_.min()


def landau_func(x, loc, scale, amplitude, baseline):
    return amplitude * moyal.pdf(x, loc=loc, scale=scale) + baseline


def estimate_baseline_landau(signal):
    # Initial parameter guess
    max_index = np.argmax(signal)
    loc_guess = max_index
    half_max = (signal[max_index] + signal.min()) / 2
    left_index = np.where(signal[:max_index] <= half_max)[0]
    right_index = np.where(signal[max_index:] <= half_max)[0] + max_index

    if (len(left_index) > 0) and (len(right_index) > 0):
        scale_guess = (right_index[0] - left_index[-1]) / 2
    else:
        scale_guess = 5
    amplitude_guess = signal[max_index]
    baseline_guess = signal.min()
    initial_guess = [loc_guess, scale_guess, amplitude_guess, baseline_guess]

    # Fit the noisy waveform
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            popt, _ = curve_fit(landau_func, np.arange(len(signal)), signal, p0=initial_guess)
        except RuntimeError:
            return np.median(signal)

    # Extract the estimated baseline
    return popt[-1]