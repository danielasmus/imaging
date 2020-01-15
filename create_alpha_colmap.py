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
import matplotlib as mpl
from matplotlib.colors import colorConverter

def create_alpha_colmap(mincol='black',maxcol='black', minalpha=0, maxalpha=1):
    """
    create a matplot lib colormap with a gradient in alpha and/or color
    """


    # generate the colors for your colormap
    color1 = colorConverter.to_rgba(mincol)
    color2 = colorConverter.to_rgba(maxcol)

    # make the colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)
    cmap._init() # create the _lut array, with rgba values

    # create your alpha array and fill the colormap with them.
    # here it is progressive, but you can create whathever you want
    alphas = np.linspace(minalpha, maxalpha, cmap.N+3)
    cmap._lut[:,-1] = alphas

    return(cmap)

