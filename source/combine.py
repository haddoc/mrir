#!/usr/bin/env python
"""
Combine methods for multi-channel MRI data
"""
import numpy as np
import itertools
from scipy import interpolate


class CoilCombo(object):
    """Class that handles various cases for coil combine"""

    def __init__(self, **kwargs):
        """Initialize combine parameters"""
        # Patch size for adaptive array
        self.size_patch = kwargs.get("size_patch")
        self.b1_map = None
        # 3D matrix size of final imaging data
        self.matrix_size = kwargs.get("matrix_size")
        # Noise data used for prewhitening
        self.noise_data = kwargs.get("noise_data")
        self.cov_noise = None
        self.prewhite_transform = None

    def compute_prewhite(self, data_in):
        """Derive the prewhitening matrix and apply it to data"""
        if data_in is None:
            self.cov_noise = None
            self.prewhite_transform = None
            return
        n_line, n_coil, n_read = data_in.shape
        data_coil = np.transpose(data_in, [0, 2, 1])
        data_coil = np.reshape(data_coil, [n_line * n_read, n_coil])
        cov_mat = np.dot(np.conj(data_coil.T), data_coil)
        # Use a Cholesky matrix for a efficient inverse.
        self.cov_noise = cov_mat
        self.prewhite_transform = np.linalg.inv(np.linalg.cholesky(cov_mat))

    def compute_b1(self, data_in, cov_noise=None):
        """Extract complex coil sensitivity maps from data
        :arg data_in of dimensions (nx, ny, nz, nc) with nc number of channels
        :kwarg cov_noise: covariance matrix of noise-only data
        """
        size_patch = np.array([4, 4, 4])
        if self.size_patch is not None:
            size_patch = np.array(self.size_patch)
        data_coil = data_in.copy()
        size_im = data_coil.shape[:-1]
        n_patch = np.ceil(size_im / size_patch).astype(int)
        n_channel = data_in.shape[-1]
        # Get the noise covariance
        if cov_noise is None:
            cov_noise = np.eye(n_channel)
        # rephase data using coil complex average
        mean_phase = np.angle(np.mean(data_coil, axis=-1))
        data_coil = data_coil * np.exp(-1j * np.tile(mean_phase[..., np.newaxis], [1, 1, 1, n_channel]))
        # Interpolate image to a multiple size of patch def
        size_im_r = n_patch * size_patch
        for dim in range(3):
            if size_im_r[dim] != size_im[dim]:
                print("Interpolate dimension {} from {} to {}".format(dim, size_im[dim], size_im_r[dim]))
                grid_i = np.linspace(0, 1, size_im[dim])
                grid_o = np.linspace(0, 1, size_im_r[dim])
                f_interp = interpolate.interp1d(grid_i, data_coil, axis=dim)
                data_coil = f_interp(grid_o)
        # Reshape the data into patches
        data_patch = np.reshape(
            data_coil, 
            (n_patch[0], size_patch[0], n_patch[1], size_patch[1], n_patch[2], size_patch[2], n_channel))
        data_patch = np.transpose(data_patch, (0, 2, 4, 1, 3, 5, 6))
        print("Size of patchy data {}".format(data_patch.shape))
        # Compute matched filter in each patch
        b1_patch = np.zeros(np.hstack((n_patch, n_channel)), dtype=np.complex64)
        cov_noise_inv = np.linalg.inv(cov_noise)
        for ix, iy, iz in itertools.product(range(n_patch[0]), range(n_patch[1]), range(n_patch[2])):
            im_patch = np.squeeze(data_patch[ix, iy, iz, ...])
            im_patch = np.reshape(im_patch, (np.prod(size_patch), n_channel))
            cov_signal = np.dot(im_patch.T, np.conj(im_patch))
            eig_values, eig_basis = np.linalg.eig(np.dot(cov_noise_inv, cov_signal))
            m_opt = eig_basis[:, np.argmax(np.abs(eig_values))]
            m_opt /= np.sqrt(np.dot(np.conj(m_opt).T, np.dot(cov_noise_inv, m_opt)))
            b1_patch[ix, iy, iz, :] = m_opt
        # Interpolate patch b1 to image size
        print("interpolate to initial size")
        samples = [np.linspace(0, 1, _n) for _n in n_patch]
        interp_points = np.mgrid[
            0:1:(size_im[0] * 1j),
            0:1:(size_im[1] * 1j),
            0:1:(size_im[2] * 1j)]
        interp_points = np.transpose(interp_points,(1,2,3,0))
        b1_map = np.zeros(data_in.shape, dtype=np.complex64)
        for ch in range(n_channel):
            b1_map[..., ch] = interpolate.interpn(samples, b1_patch[..., ch], interp_points, 'nearest')
        self.b1_map = b1_map

    def forward(self, data_in, data_noise=None, flag_prewhite=False, mode='adaptive'):
        """Transform multi-channel data into single image data
        :arg data_in: array of multi-channel 3D data (nx, ny, nz, nc)
        :kwarg flag_prewhite: if True, apply the prewhitening transform to data prior to combining
        :kwarg mode: how channels are combined. 'SoS': sum-of-square, 'adaptive': matched filter
        """
        assert isinstance(data_in, np.ndarray)
        assert (data_in.ndim == 4)
        data_coil = data_in.copy()
        nx, ny, nz, nc = data_in.shape
        self.compute_prewhite(data_noise)
        if flag_prewhite and self.prewhite_transform is not None:
            data_coil = np.reshape(data_coil, [nx * ny * nz, nc])
            data_coil = np.dot(data_coil, self.prewhite_transform)
            data_coil = np.reshape(data_coil, [nx, ny, nz, nc])
            self.cov_noise = None  # we don't want to apply prewhitening during combine step

        # Sum of square combo
        if mode == 'SoS':
            im_combo = np.sqrt(np.sum(np.abs(data_coil) ** 2, axis=-1))
        else:
            self.compute_b1(data_coil, cov_noise=self.cov_noise)
            assert (self.b1_map is not None)
            im_combo = np.sum(np.conj(self.b1_map) * data_coil, axis=-1)
        return im_combo


if __name__ == "__main__":
    im_4d = np.load("/Users/had-mr/github/mrir/notebooks/image_coil.npy")
    comb = CoilCombo(matrix_size=im_4d.shape[:-1])
    im_3d = comb.forward(im_4d)
    import matplotlib.pyplot as plt
    plt.imshow(np.abs(im_3d[:,:,12]), cmap='gray')
    # plt.imshow(np.abs(comb.b1_map[:,:,17,3]), cmap='gray')
    plt.show()