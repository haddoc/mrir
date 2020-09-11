import numpy as np
import scipy.signal as sps

def compute_welch(data_time, bandwidth, dim_channel):
    """Compute power spectrum using welch method
    :arg data_time: n-dimension array of time domain data
    :arg bandwidth: acquisition bandwidth (float)
    :arg dim_channel: which dimension the power spectrum is computed on
    """
    shape_data = list(data_time.shape)
    shape_channel = [shape_data[dim_channel]]
    dims_new = [dim_channel]
    for _i, _d in enumerate(shape_data):
        if _i != dim_channel:
            shape_channel.append(_d)
            dims_new.append(_i)
    data_channel = np.transpose(data_time, dims_new)
    data_channel = np.reshape(data_channel, (shape_channel[0], np.prod(shape_channel[1:])))
    # dummy run to get dimensions of power spectrum
    _, noise_power = sps.welch(data_channel[0, :], fs=bandwidth, return_onesided=False)
    noise_power = noise_power[1:-1]
    # Get spectrum for all channels
    power_spectrum = np.zeros((data_channel.shape[0], noise_power.size))
    for _ch in range(data_channel.shape[0]):
        # Complex data: one-sided
        axis_freq, noise_power = sps.welch(data_channel[_ch, :], fs=bandwidth, return_onesided=False)
        # remove edge samples (filter effect)
        axis_freq = axis_freq[1:-1]
        noise_power = noise_power[1:-1]
        # Sort in incremental axis_frequency
        idx_sort = np.argsort(axis_freq)
        axis_freq = axis_freq[idx_sort]
        noise_power = noise_power[idx_sort]
        power_spectrum[_ch, :] = noise_power

    return axis_freq, power_spectrum