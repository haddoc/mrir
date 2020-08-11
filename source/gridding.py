"""
Fourier reconstruction of k-space data with non-uniform sampling
"""
import numpy as np
from pynufft import NUFFT_cpu


class NuFFT(object):

    def __init__(self, k_coords=None, image_size=None):
        """Initialize the transforms"""
        kernel_size = (6, 6)  # interpolation size
        image_size = np.array(image_size, dtype=int)
        k_coords = np.array(k_coords, dtype=float)
        kspace_size = (1.5 * image_size).astype(int)
        self.nufft_obj = NUFFT_cpu()
        self.nufft_obj.plan(k_coords, tuple(image_size), tuple(kspace_size), kernel_size)

    def compute_density(self, samples_2d=None):
        """Compute the sampling density of a nonuniform 2D sampling"""
        pass

    def forward(self, data_in):
        """Transform nonuniform-sample data from k-space to image space"""
        return self.nufft_obj.forward(data_in)

    def adjoint(self, data_in):
        """Transform nonuniform-sample data from k-space to image space"""
        self.nufft_obj.xx2x
        return self.nufft_obj.adjoint(data_in)
