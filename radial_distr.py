#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.1"

"""
HISTORY:
    - 2020-01-15: created by Daniel Asmus
    - 2020-01-15: cleaned up packages


NOTES:
    -

TO-DO:
    -
"""


import numpy as np
# import scipy.stats
from scipy.stats import binned_statistic


def radial_distr(im, cen=None, samp=None, maxrad=None, exclude_empty_bins=True,
                 binmeth='median'):
    """
    compute the radial distribution of an image for a optionally given centre
    position and maxiumum radius whereas samp controls the radial binsize and
    binmeth the method of averaging the values within a ring bin.
    """

    siz = np.array(np.shape(im))

    if cen is None:
        cen = 0.5*siz

    r = np.zeros(siz[0] * siz[1])
    oval = np.zeros(siz[0] * siz[1])
    z = 0
    for x in range(siz[1]):
        for y in range(siz[0]):
            r[z] = np.sqrt((x - cen[1])**2 + (y - cen[0])**2)
            oval[z] = im[y, x]
            z = z + 1

    # print(np.min(r), np.max(r), xcen, ycen)

    # --- sort values by radius
    id = np.argsort(r)
    r = r[id]
    oval = oval[id]

    # --- in case of radial binning:
    if samp is not None:

        if maxrad is None:
            maxrad = np.max(r)

        bins = np.linspace(0, maxrad, num=samp)

        bvals = binned_statistic(r, oval, statistic=binmeth, bins=samp,
                                 range=(0, maxrad))[0]

        if exclude_empty_bins:
            r = bins[np.isfinite(bvals)]
            oval = bvals[np.isfinite(bvals)]
        else:
            r = bins
            oval = bvals

    return(r, oval)

