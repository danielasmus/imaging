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


def extend_image(im, fac=None, addx=None, addy=None):
    """
    Extend a given image by adding additional pixels around its borders
    """
    s = np.array(np.shape(im))

    if fac is not None:
        newsize = np.array(np.round(s * fac), dtype=int)
    elif addx is not None or addy is not None:
        newsize = np.array(np.round(s + [addy, addx]), dtype=int)
    else:
        raise ValueError('Invalid Inputs')

    newim = np.zeros(newsize)

    newim[int(0.5*(newsize[0] - s[0])) : int(0.5*(newsize[0] + s[0])),
          int(0.5*(newsize[1] - s[1])) : int(0.5*(newsize[1] + s[1]))] = im

    return(newim)
