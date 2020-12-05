import numpy as np
import scipy.signal as sps


def compute_spectrogram(data_time, bandwidth):
    """Compute spectral density for each acquisition, then average along repetitions"""
    n_channel, n_read, n_lines = data_time.shape
    # Dummy run to get spectrogram size
    coords_freq, spec_0 = sps.welch(data_time[0,:,0], fs=bandwidth, return_onesided=False, detrend=False)
    reord = np.argsort(coords_freq)
    coords_freq = coords_freq[reord]
    # Compute spectrogram for each channel and line
    power_spectrum = np.zeros((n_channel, spec_0.size, n_lines))
    for _i in range(n_lines):
        for _ch in range(n_channel):
            _, spec = sps.welch(data_time[_ch,:,_i], fs=bandwidth, return_onesided=False, detrend=False)
            power_spectrum[_ch,:,_i] = spec[reord]
    power_spectrum = np.squeeze(np.mean(power_spectrum, axis=-1))
    return coords_freq, power_spectrum


def compute_welch(data_time, bandwidth):
    """Compute power density along the readout x line dimension"""
    n_channel = data_time.shape[0]
    data_coils = np.reshape(data_time, [n_channel, -1])
    # Remove zero (from masking)
    data_coils = data_coils[:, np.abs(data_coils[0, :]) > 0]
    # Dummy to get dimensions and reorder
    coords_freq, spec_0 = sps.welch(data_coils[0,:], fs=bandwidth, return_onesided=False, detrend=False)
    reord = np.argsort(coords_freq)
    coords_freq = coords_freq[reord]
    # Compute spectrum for each channel
    power_spectrum = np.zeros((n_channel, spec_0.size))
    for _ch in range(n_channel):
        _, spec = sps.welch(data_coils[_ch, :], fs=bandwidth, return_onesided=False, detrend=False)
        power_spectrum[_ch,:] = spec[reord]
    return coords_freq, power_spectrum


def compute_kspace_weights(k_coords=None, n_readout=None, n_channels=None, threshold=None):
    """Compute signal weighing for a 3D Cartesian multichannel acquisision, given 2D k-space coordinates
    :returns data weights as dimensiong (lines, channels, readout)
    """
    if k_coords is None or n_readout is None or n_channels is None:
            return 1
    # Weight function in readout (assumes echo at the center)
    weight_readout = np.abs(np.linspace(-1, 1, n_readout))
    if isinstance(threshold, float):
        weight_readout = np.array(weight_readout > threshold, dtype=float)
    else:
        weight_readout = 1 - np.exp(-5 * weight_readout ** 2)
    # Weight function on line encoding
    k_norm = np.sqrt(np.sum(k_coords ** 2, axis=-1))
    k_norm /= np.max(k_norm)
    if isinstance(threshold, float):
        weight_coords = np.array(k_norm > threshold, dtype=float)
    else:
        weight_coords = 1 - np.exp(-5 * k_norm ** 2)
    # Combine all weights
    weights_all = np.dot(weight_readout[:, np.newaxis], weight_coords[:, np.newaxis].T)
    # Add multi-channel dimension to weights (identical weights on all channels)
    weights_all = np.tile(weights_all[:, :, np.newaxis], n_channels)
    # Set the weights as dimensions (lines, channels, readout)
    k_weights = np.transpose(weights_all, [2, 0, 1])
    return k_weights
