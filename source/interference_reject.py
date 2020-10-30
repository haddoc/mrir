"""
Interference rejection from MRI data
"""
import numpy as np
from source.utils import compute_kspace_weights

class InterferenceReject(object):
    """Class for handling various cases of interference rejection"""

    def __init__(self, **kwargs):
        """Initialize with options passed as kwargs"""
        # Selection index for NMR channels (list type)
        self.channels_signal = kwargs.get("channels_signal")
        # Selection index for Reference channels (list type)
        self.channels_noise = kwargs.get("channels_noise")

    def apply(self, data_signal=None, data_noise=None):
        """Perform interference rejection
        Assumes Cartesian data, and performs spectral interference removal on the 3rd dimension
        :kwarg data_signal: the main imaging scan data 
            time domain data of dimensions (channels, readout, lines)
        :kwarg data_noise: (optional) prescan data acquired without signal 
            time domain of dimensions (channels, readout, lines)
        """
        assert isinstance(data_signal, np.ndarray)
        assert isinstance(data_noise, np.ndarray)

        # Get scan data to spectral domain
        spec_scan = np.fft.fft(data_signal, axis=1)

        # Get noise data to spectral domain
        spec_noise = np.fft.fft(data_noise, axis=1)

        # Compute coefficients using noise scan
        coeff_all = self.compute_coeffs(spec_noise)

        # Perform rejection
        num_freq_bins = spec_scan.shape[1]
        ndims_output = list(spec_scan.shape)
        ndims_output[0] = len(self.channels_signal)
        spec_cor = np.zeros(ndims_output, dtype=np.complex64)
        for _f in range(num_freq_bins):
            sig = spec_scan[self.channels_signal, _f, :]
            ref = spec_scan[self.channels_noise, _f, :]
            spec_cor[:, _f, :] = sig - np.dot(coeff_all[:, :, _f], ref)

        data_cor = np.fft.ifft(spec_cor, axis=1)
        return data_cor

    def compute_coeffs(self, cal_data):
        """Compute the correction coefficients for each frequency"""
        num_freq_bins = cal_data.shape[1]
        coeff_all = np.zeros((len(self.channels_signal), len(self.channels_noise), num_freq_bins), dtype=np.complex64)
        for _f in range(num_freq_bins):
            cal_sig = cal_data[self.channels_signal, _f, :]
            cal_ref = cal_data[self.channels_noise, _f, :]
            coeff = np.linalg.lstsq(cal_ref.T, cal_sig.T, rcond=-1)[0]
            if coeff.ndim == 1:
                coeff.expand_dims(axis=-1)
            coeff_all[:, :, _f] = coeff.T
        return coeff_all
