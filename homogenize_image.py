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

from astropy.stats import sigma_clip

from .get_background import get_background as _get_background


def homogenize_image(im, sigmaclip=0, percen=99.9, bgtype='median', sigcut=0, ignore_aper=None):
    """
    homogenize an image for example for the structural similarity comparison (SSIM)
    """

    # --- determine the background
    bg, bgstd = _get_background(im, method=bgtype, ignore_aper=ignore_aper)

    outim = np.copy(im)

    # --- cut the NaNs
    outim[np.isnan(im)] = bg

    # --- sigma clipping
    if sigmaclip > 0:
        outim = sigma_clip(outim, sigma=sigmaclip, iters=2, masked=False)

    # --- subtract background
    outim = outim - bg - sigcut * bgstd
    outim[outim < 0] = 0

    # --- normalize by smoothed max
    # outim = outim/np.max(gaussian_filter(outim, sigma=normsmooth))
    # --- normalize by the high percenile (3 times faster than the gaussian filter
    outim = outim/np.percentile(outim, percen)


    return(outim)

