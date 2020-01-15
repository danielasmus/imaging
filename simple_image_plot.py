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
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

from .get_background import get_background as _get_background


def simple_image_plot(im, fout, log=False, percentile=1, maxbg=False,
                      binsize=None, cmap='gnuplot2', pwidth=14.17):

    if log:
        dim = np.log10(1000.0 * (im - np.nanmin(im)) /
                          (np.nanmax(im) - np.nanmin(im)) + 1)
        if binsize is None:
            binsize=0.003

    else:
        dim = im

    if maxbg:
        vmin = _get_background(dim, ignore_aper=None, method='distmax',
                              binsize=binsize, show_plot=False)[0]
    else:
        vmin = np.nanpercentile(dim, percentile)

    # print(vmin)
    plt.clf()

    s = np.shape(im)
    pheight = pwidth * s[0]/s[1]

    fig = plt.figure(figsize=(pwidth, pheight))


    plt.imshow(dim, origin='bottom',
               interpolation='nearest',
               vmin=vmin, cmap=cmap,
               vmax=np.nanpercentile(dim, 100-percentile))


    plt.savefig(fout, bbox_inches='tight', pad_inches=0.01)
#    plt.savefig(fout)

    plt.clf()
    plt.close(fig)

