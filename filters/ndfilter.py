##########################################
# File: ndfilter.py                      #
# Copyright Richard Stebbing 2015.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import numpy as np
import scipy.fftpack as fftpack


class NDFilter(object):
    """
    NDFilter(shape, fs=None)

    Abstract base class for n-dimensional discrete filters.

    Parameters
    ----------
    shape : int or sequence of ints
        Dimensions of the filter.

    fs : array_like optional
        Sampling frequencies along each dimension of the filter. This is
        used for the construction of the filter in the frequency domain.
    """
    def __init__(self, shape, fs=None):
        fs = np.asarray(fs) if fs is not None else np.ones(len(shape))

        # Extent is calculated so that the correct number of samples are taken
        # at the given sampling frequencies.
        del_x = 1. / fs
        self.extents = shape * del_x
        self.fs = fs

    def __del__(self):
        fftpack._fftpack.destroy_zfftnd_cache()

    def apply(self, image, freq_image=False, invert=True):
        """
        apply(image, freq_image=False, invert=True)

        Apply the current filter.

        Parameters
        ----------
        image : array_like
            Image to apply filter to.

        freq_image : bool optional
            Flag to indicate if the input image is already the DFT of the
            input image.

        invert : bool optional
            Flag to indicate if the output image should be inverted from the
            frequency domain.
        """
        try:
            H = self.H
        except AttributeError:
            raise AttributeError, 'No filter currently set.'

        if not freq_image:
            image = fftpack.fftn(image)

        E = image * H

        if invert:
            E = np.real(fftpack.ifftn(E))

        return E

def zero_centred_grid(shape, extents=None):
    """
    zero_centred_grid(shape, extents=None)

    Create an n-dimensional zero-centred grid.

    Parameters
    ----------
    shape : tuple
        Number of samples along each dimension.
    extents : tuple optional
        Length to sample over along each dimension.
    """
    # Number of dimensions.
    ndim = len(shape)

    if extents is None:
        extents = shape

    # Form periodic grid in [0,e) across each dimension.
    slices = [slice(0, e, complex(0, dim + 1))
              for e, dim in zip(extents, shape)]
    grid = np.mgrid[slices]

    every = [slice(None, None, None)]
    slices = tuple(every + [slice(None, -1) for dim in shape])
    grid = grid[slices]

    # Now zero-centre the grid and extract the sampling distance along each
    # dimension.
    dif = []
    centre = []
    for i, grid_ in enumerate(grid):
        # Extract along dimension of interest.
        slices = tuple([0]*i + every + [0]*(ndim - 1 - i))

        # Get adjustment element.
        v = grid_[slices]
        j = len(v)/2

        # Make adjustment.
        grid_ -= v[j]

        # Save sampling period and index of centre.
        if len(v) > 1:
            dif.append(v[1] - v[0])
        else:
            dif.append(0.0)
        centre.append(j)

    return grid, tuple(dif), tuple(centre)
