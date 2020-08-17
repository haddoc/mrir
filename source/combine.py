"""
Combine methods for multi-channel MRI data
"""
import numpy as np


class CoilCombo(object):
    """Class that handles various cases for coil combine"""

    def __init__(self, **kwargs):
        """Initialize combine parameters"""
        self.sense_map = None
        # 3D matrix size of final imaging data
        self.matrix_size = kwargs.get("matrix_size")
        # Noise data used for prewhitening
        self.noise_data = kwargs.get("noise_data")
        self.prewhite_transform = None

    def prewhite(self):
        """Derive the prewhitening matrix and apply it to data"""
        if self.noise_data is None:
            return
        n_line, n_coil, n_read = self.noise_data.shape
        data_coil = np.transpose(self.noise_data, [0, 2, 1])
        data_coil = np.reshape(data_coil, [n_line * n_read, n_coil])
        corr_mat = np.dot(np.conj(data_coil.T), data_coil)
        # Use a Cholesky matrix for a efficient inverse.
        self.prewhite_transform = np.linalg.inv(np.linalg.cholesky(corr_mat))
        print(self.prewhite_transform.shape)
        

    def forward(self, data_in, flag_prewhite=True, mode='adaptive'):
        """Transform multi-channel data into single image data"""
        assert isinstance(data_in, np.ndarray)
        assert (data_in.ndim == 4)
        nx, ny, nz, nc = data_in.shape
        if flag_prewhite and self.prewhite_transform is not None:
            data_pw = np.reshape(data_in, [nx * ny * nz, nc])
            data_pw = np.dot(data_pw, self.prewhite_transform)
            data_pw = np.reshape(data_pw, [nx, ny, nz, nc])
        else:
            data_pw = data_in
        
        # Sum of square combo
        if mode == 'SoS':
            im_sos = np.sqrt(np.sum(np.abs(data_pw) ** 2, axis=-1))
            return im_sos
        else:
            return
