##########################################
# File: monogenic.py                     #
# Copyright Richard Stebbing 2015.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import numpy as np
import scipy.fftpack as fftpack

from filters.ndfilter import NDFilter, zero_centred_grid


class RieszTransform(NDFilter):
    """
    RieszTransform(shape, fs=None)

    Filter to calculate the Riesz transform components.

    Parameters
    ----------
    Refer to NDFilter.
    """
    def __init__(self, shape, fs=None):
        NDFilter.__init__(self, shape, fs)

        u, del_u, u_centre = zero_centred_grid(shape, self.fs)

        with np.errstate(all='ignore'):
            H = u/np.sqrt(np.sum(u**2, axis=0))

        H[(slice(None), ) + u_centre] = 0

        # Make complex and reorder (except on first axis).
        H = H.astype(complex) * complex(0, 1)
        self.H = fftpack.ifftshift(H, axes=range(1, H.ndim))

    def apply(self, image, freq_image=False, invert=True):
        if not freq_image:
            image = fftpack.fftn(image)

        if invert:
            dtype = np.float
        else:
            dtype = np.complex

        result = np.empty((len(self.H),) + image.shape, dtype=dtype)

        for i, H in enumerate(self.H):
            result[i, ...] = (np.real(fftpack.ifftn(image*H)) if invert
                              else image * H)
        return result

    apply.__doc__ = NDFilter.apply.__doc__

class FeatureAsymmetry(RieszTransform):
    """
    FeatureAsymmetry(shape, fs=None)

    Filter to calculate the feature asymmetry measure over an image.

    Parameters
    ----------
    Refer to NDFilter.
    """
    def __init__(self, shape, fs=None):
        RieszTransform.__init__(self, shape, fs)

    def apply(self, image, T=1.0, orientation=False, freq_image=False,
              return_riesz=False):
        """
        apply(image, T=1.0, orientation=False, freq_image=False,
              return_riesz=False)

        Apply the feature asymmetry measure to the image.

        Parameters
        ----------
        image : array_like
            Image to apply filter to.

        T : float optional
            Threshold to remove noise components, expressed as a multiple of
            the geometric mean.

        orientation : bool optional
            Calculate and return the orientation image as well.

        freq_image : bool optional
            Flag to initiate if the input image is already the DFT of the
            input image.

        return_riesz : bool optional
            Return the Riesz components as well.
        """
        riesz = RieszTransform.apply(self, image, freq_image)

        # Invert only if required (even component is required).
        if freq_image:
            image = np.real(fftpack.ifftn(image))

        # Calculate 'odd' component.
        odd = np.sqrt(np.sum(riesz**2, axis=0))

        # Calculate numerator and denominator for FA.
        N = np.abs(odd) - np.abs(image)
        D = np.sqrt(odd**2 + image**2) + 1e-6

        # Store the ratio.
        self.R1 = N / D

        # Get base threshold (geometric mean of the denominator).
        T_base = np.exp(np.mean(np.log(D)))
        self.R2 = T_base / D

        # Apply the threshold to generate the FA image (self.E).
        E = self.apply_new_threshold(T)

        # Calculate orientation if required.
        ret = [E]

        if orientation:
            O = np.rollaxis(riesz, 0, riesz.ndim)
            ret.append(O)

        if return_riesz:
            ret.append(riesz)

        return ret[0] if len(ret) == 1 else tuple(ret)

    def apply_new_threshold(self, T):
        """
        apply_new_threshold(T)

        If apply has been called already, then try a different threshold. This
        method avoids complete recalculation of the Riesz components etc.

        Parameters
        ----------
        T : float
            Multiple of the geometric mean of the denominator calculated in the
            feature asymmetry measure.
        """
        E = self.R1 - T * self.R2
        E[E < 0.0] = 0.0
        return E

def main():
    import matplotlib.pyplot as plt
    from filters.log_gabor import IsotropicLogGabor

    # Example: An image with twice the number of samples along the first axis
    # (with twice the sampling frequency).
    sp = np.array([0.5, 1.0])
    im = np.zeros((200, 100), dtype=np.float64)
    im[60:200 - 60, 30:100 - 30] = 1.0
    im[80:200 - 80, 40:100 - 40] = 0.0

    # Filter with an isotropic log-Gabor filter of various centre
    # frequencies ...
    fs = 1.0 / sp
    filter_ = IsotropicLogGabor(im.shape, fs=fs)
    im_lg = []
    for f0 in [0.02, 0.03, 0.04]:
        filter_.filter(f0, 2.0)
        im_lg.append(filter_.apply(im))

    # ... and apply feature asymmetry.
    filter_ = FeatureAsymmetry(im.shape, fs=fs)
    im_fa = [filter_.apply(im_) for im_ in im_lg]

    f, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(im, cmap='gray', interpolation='none')
    axs[0, 0].set_aspect(sp[0] / sp[1])
    for i, im_ in enumerate(im_fa, start=1):
        ax = axs[i / 2, i % 2]
        ax.imshow(im_, cmap='gray', interpolation='none')
        ax.set_aspect(sp[0] / sp[1])

    plt.show()

if __name__ == '__main__':
    main()
