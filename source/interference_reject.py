"""
Interference rejection from MRI data
"""
import numpy as np

class InterferenceReject(object):
    """Class for handling various cases of interference rejection"""

    def __init__(self, **kwargs):
        """Initialize with options passed as kwargs"""
        # Are we only using a prescan for computing coefficients
        self.flag_use_prescan = kwargs.get("use_prescan", False)
        # Are we using k-space weighting for computing coefficients
        self.flag_use_weights = kwargs.get("use_weights", False)
        # How many channels in the NMR coil array
        self.num_channels_signal = kwargs.get("channels_signal", 1)
        # How many total channels in the ADC
        self.num_channels_all = kwargs.get("channels_all", 16)
        # Initialize attributes
        self.k_weights = None

    def compute_weights(self, num_readout_points=None):
        """Compute the k-space data weighting"""
        if self.k_coords is None or num_readout_points is None:
            return
        # Weight function in readout (assumes echo at the center)
        weight_readout = np.linspace(0, 1, num_readout_points) - 0.5
        weight_readout = 1 - np.exp(-10 * weight_readout ** 2)
        # Weight function on line encoding
        k_norm = np.sqrt(self.k_coords[:, 0] ** 2 + self.k_coords[:, 1] ** 2)
        weight_coords = 1 - np.exp(-10 * k_norm ** 2)
        # Combine all weights
        weights_all = np.dot(weight_readout[:, np.newaxis], weight_coords[:, np.newaxis].T)
        # Add multi-channel dimension to weights (identical weights on all channels)
        weights_all = np.tile(weights_all[:, :, np.newaxis], self.num_channels_all)
        # Set the weights as dimensions (lines, channels, readout)
        self.k_weights = np.transpose(weights_all, [1, 2, 0])

    def apply(self, scan_raw=None, prescan_raw=None, k_coords=None):
        """Perform interference rejection
        Assumes Cartesian data, and performs spectral interference removal on the 3rd dimension
        :kwarg scan_raw: the main imaging scan data 
            time domain (lines, channels, readout)
        :kwarg prescan_raw: (optional) prescan data acquired without signal 
            time domain (lines, channels, readout)
        """
        assert isinstance(scan_raw, np.ndarray)
        # Compute encoding weights
        if k_coords is not None:
            self.k_coords = k_coords
            self.compute_weights(num_readout_points=scan_raw.shape[-1])
        # Get scan data to spectral domain
        spec_scan = np.fft.fft(scan_raw, axis=-1)
        
        # Get prescan data to spectral domain
        if prescan_raw is not None:
            spec_prescan = np.fft.fft(prescan_raw, axis=-1)
        else:
            spec_prescan = None
            self.flag_use_prescan = False
        
        # Compute coefficients using the prescan only
        if self.flag_use_prescan:
            coeff_all = self.compute_coeffs(spec_prescan)
        elif self.flag_use_weights and self.k_weights is not None:
            coeff_all = self.compute_coeffs(spec_scan * self.k_weights)
        else:
            coeff_all = self.compute_coeffs(spec_scan)
        
        # Selection indices for signal channels and reference channels
        sel_chan_sig = slice(self.num_channels_signal)
        sel_chan_ref = slice(self.num_channels_signal, self.num_channels_all)

        # Perform rejection
        num_freq_bins = spec_scan.shape[-1]
        ndims_corrected = list(spec_scan.shape)
        ndims_corrected[1] = self.num_channels_signal
        spec_cor = np.zeros(ndims_corrected, dtype=np.complex64)
        for _f in range(num_freq_bins):
            sig = spec_scan[:, sel_chan_sig, _f]
            ref = spec_scan[:, sel_chan_ref, _f]
            spec_cor[:, :, _f] = sig - np.dot(ref, coeff_all[:, :, _f])

        data_cor = np.fft.ifft(spec_cor, axis=-1)
        return data_cor

    def compute_coeffs(self, cal_data):
        """Compute the correction coefficients for each frequency"""
        num_channels_noise = self.num_channels_all - self.num_channels_signal
        num_freq_bins = cal_data.shape[-1]
        coeff_all = np.zeros((self.num_channels_signal, num_channels_noise, num_freq_bins), dtype=np.complex64)
        sel_chan_sig = slice(self.num_channels_signal)
        sel_chan_ref = slice(self.num_channels_signal, self.num_channels_all)
        for _f in range(num_freq_bins):
            cal_sig = cal_data[:, sel_chan_sig, _f]
            cal_ref = cal_data[:, sel_chan_ref, _f]
            coeff = np.linalg.lstsq(cal_ref, cal_sig, rcond=-1)[0]
            coeff_all[:, :, _f] = coeff
        return coeff_all
