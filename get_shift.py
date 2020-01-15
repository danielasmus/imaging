#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-15: created by Daniel Asmus


NOTES:
    -

TO-DO:
    -
"""


import numpy as np
import sys

from .crop_image import crop_image as _crop_image
from .get_pointsource import get_pointsource as _get_pointsource



def get_shift(im, refpos=None, refim=None, guesspos=None, searchbox=None,
              fitbox=None, method='fast', guessmeth='max', searchsmooth=3,
              refsign=1, guessFWHM=None, guessamp=None, guessbg=None,
              fixFWHM=False, minamp=0.01, maxamp=None, maxFWHM=None,
              minFWHM=None, verbose=False, plot=False, silent=False):

    """
    detect and determine the shift or offset of one image either with respect
    to a reference image or a reference position. In the first case, the
    shift is determined with cross-correlation, while in the second a Gaussian
    fit is performed
    the convention of the shift is foundpos - refpos
    OPTIONAL INPUT
    - refpos: tuple, given (approx.) the reference position of a source that
    can be used for fine centering before cropping
    - method: string(max, mpfitgauss, fastgauss, skimage, ginsberg), giving the method that
    should be used for the fine centering using the reference source.
    'skimage' and 'ginsberg' do 2D cross-correlation
    - refim: 2D array, giving a reference image if the method for the
    centering is cross-correlation
    - fitbox: scalar, giving the box length in x and y for the fitting of the
    reference source

    """

    s = np.array(np.shape(im))

    if guesspos is None and refpos is not None:
        guesspos = refpos
    elif refpos is None and guesspos is not None:
        refpos = guesspos
    elif guesspos is None and refpos is None:
        guesspos = 0.5*np.array(s)
        refpos = guesspos

    if verbose:
        print("GET_SHIFT: guesspos: ", guesspos)
        print("GET_SHIFT: refpos: ", refpos)


    # --- if a reference image was provided then perform a cross-correlation
    if method in ['cross', 'skimage', 'ginsberg']:


        sr = np.array(np.shape(refim))
        if verbose:
            print("GET_SHIFT: input image dimension: ", s)
            print("GET_SHIFT: reference image dimension: ", sr)

        # --- for the cross-correlation, the image and reference must have the
        #     the same size
        if (s[0] > sr[0]) | (s[1] > sr[1]):
            cim = _crop_image(im, box=sr, cenpos=guesspos)

            cenpos = guesspos

#            # --- adjust the ref and guesspos
#            refpos = refpos - guesspos + 0.5 * sr
#            guesspos = 0.5 * sr

            if verbose:
                print("GET_SHIFT: refim smaller than im --> cut im")
                print("GET_SHIFT: adjusted guesspos: ", guesspos)
                print("GET_SHIFT: adjusted refpos: ", refpos)


        elif (sr[0] > s[0]) | (sr[1] > s[1]):
            cim = im
            refim = _crop_image(refim, box=s)

            cenpos = 0.5 * s

        else:
            cim = im

            cenpos = 0.5 * s


        # --- which cross-correlation algorithm should it be?
        if (method == 'cross') | (method == 'skimage'):
            from skimage.feature import register_translation

            shift, error, diffphase = register_translation(cim, refim,
                                                           upsample_factor=100)

            #print(shift,error)
            error = [error, error]  # apparently only on error value is returned?

        elif method == 'ginsberg':
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                import image_registration
            np.seterr(all='ignore') # silence warning messages about div-by-zero

            dx, dy, ex, ey = image_registration.chi2_shift(refim, cim,
                                                           return_error=True,
                                                           zeromean=True,
                                                           upsample_factor='auto')

            shift = [dy, dx]
            error = [ey, ex]

        else:
            print("GET_SHIFT: ERROR: requested method not available: ", method)
            sys.exit(0)

        fitim = None
        params = None
        perrs = None

        # --- correct shift for any difference between guesspos and refpos:
        shift = [shift[0] + cenpos[0] - refpos[0],
                 shift[1] + cenpos[1] - refpos[1]]


   # --- if not reference image is provided then perform a Gaussian fit
    else:

        # --- fit a Gaussian to find the center for the crop
        params, perrs, fitim = _get_pointsource(im, searchbox=searchbox,
                                               fitbox=fitbox, method=method,
                                               verbose=verbose,
                                               guesspos=guesspos, sign=refsign,
                                               plot=plot, guessFWHM=guessFWHM,
                                               guessamp=guessamp,
                                               guessbg=guessbg,
                                               searchsmooth=searchsmooth,
                                               fixFWHM=fixFWHM, minamp=minamp,
                                               maxFWHM=maxFWHM,
                                               minFWHM=minFWHM,
                                               silent=silent)

        # --- compute the shift from the fit results:
        shift = np.array([params[2] - refpos[0], params[3] - refpos[1]])

        # --- eror on the shift from the fit results:
        error = np.array([perrs[2], perrs[3]])


    if verbose:
        print('GET_SHIFT: Found shift: ', shift)
        print('GET_SHIFT: Uncertainty: ', error)
        print('GET_SHIFT: Fit Params: ', params)

    return(shift, error, [params, perrs, fitim])

