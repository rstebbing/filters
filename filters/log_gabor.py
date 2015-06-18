##########################################
# File: log_gabor.py                     #
# Copyright Richard Stebbing 2015.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import warnings

import numpy as np
import scipy.fftpack as fftpack

from filters.ndfilter import NDFilter, zero_centred_grid


def k_const(num_octaves):
    """
    k_const(num_octaves)

    Calculate the wave-shape constant to achieve the required bandwidth.

    Parameters
    ----------
    num_octaves :
        Bandwidth of the log-Gabor filter.
    """
    return 2**(-num_octaves / (2.0 * np.sqrt(np.log(4))))

def log_gabor(f, k):
    """
    log_gabor(f, k)

    Log-Gabor function.

    Parameters
    ----------
    f : array_like
        Normalised frequency (relative to centre frequency of log-Gabor
        filter).

    k : float
        Wave-shape coefficient for log-Gabor filter.
    """
    return np.exp(-np.log(f)**2 / (2*np.log(k)**2))

def log_gabor_coefficients(f0, k, N, eps=1e-2):
    """
    log_gabor_coefficients(f0, k, N, eps=1e-2)

    Calculate the even and odd Gabor filter coefficients.

    Parameters
    ----------
    f0 : float
        Centre frequency.

    k : float
        Wave-shape coefficient for log-Gabor filter.

    N : int
        Number of filter coefficients.

    eps : float
        Maximum filter response before aliasing is considered.
    """
    f = np.linspace(0, 1, N)

    X = log_gabor(f[1:] / f0, k)

    # Check last term of frequency response.
    if X[-1] >= eps:
        warnings.warn('Warning: X(f) >= %f at f==1: ' % eps, X[-1],
                      RuntimeWarning)

    # Construct full single-sided response.
    X = np.hstack(([0], 2 * X, np.zeros(N - 1)))

    x = np.fft.ifft(X)
    x = np.hstack((x[-(N - 1)/2:], x[:(N - 1)/2 + 1]))

    return np.real(x), np.imag(x)

class IsotropicLogGabor(NDFilter):
    """
    IsotropicLogGabor(shape, fs=None)

    Isotropic log-Gabor filter.

    Parameters
    ----------
    Refer to NDFilter.
    """
    def __init__(self, shape, fs=None):
        NDFilter.__init__(self, shape, fs)

        # Calculate magnitude along u, v, ... plane.
        u, del_u, self.u_centre = zero_centred_grid(shape, self.fs)

        self.hyp = np.sqrt(np.sum(u**2, axis=0))

    def filter(self, f0, num_octaves):
        """
        filter(f0, num_octaves)

        Set the current filter parameters.

        Parameters
        ----------
        f0 : float
            Centre frequency of the log-Gabor filter.

        num_octaves :
            Number of octaves for the half-power bandwidth.
        """
        k = k_const(num_octaves)

        with np.errstate(all='ignore'):
            H = log_gabor(self.hyp / f0, k)
            H[self.u_centre] = 0.0

        self.H = fftpack.ifftshift(H)
        return self.H
