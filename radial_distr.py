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
import time
#import traceback
# import pdb
from collections import OrderedDict
from collections import namedtuple
from scipy import ndimage
import scipy.signal
# import scipy.stats
from scipy.stats import binned_statistic
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
from matplotlib.colors import colorConverter
from matplotlib.patches import Arc
from mpl_toolkits.axes_grid1 import ImageGrid
# from IPython import embed

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Column

from astropy.stats import sigma_clip

from astropy.modeling import models, fitting
from astropy.modeling import Fittable2DModel, Parameter

from gaussfitter import gaussfit

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

