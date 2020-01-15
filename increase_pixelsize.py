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
import time


# ---define a helper routine to get the contributions of the old pixels to
#    a given new pixel in one dimension
def get_contribution(difs, oldpfov, newpfov):

    n = len(difs)

    contr = np.zeros(n, dtype=float)

    ids = (difs > 0) & (difs < oldpfov)
    contr[ids] = 1.0 - difs[ids]/oldpfov

    ids = (difs + newpfov > 0) & (difs + newpfov < oldpfov)
    contr[ids] = (newpfov + difs[ids])/oldpfov

    ids = (difs <= 0) & (difs >= oldpfov - newpfov)
    contr[ids] = 1

#        for i in range(n):
#
#            # --- old pixel completely within new pixel
#            if 0 >= difs[i] >= (oldpfov - newpfov):
#                contr[i] = 1.0
#
#            # --- old pixel end inside new pixel
#            elif 0 < difs[i] < oldpfov:
#                contr[i] = 1.0 - difs[i]/oldpfov
#
#            # --- old pixel beginning inside new pixel
#            elif 0 < difs[i] + newpfov < oldpfov:
#                contr[i] = (newpfov + difs[i])/oldpfov

    return(contr)


def increase_pixelsize(im, oldpfov=None, newpfov=None, newsize=None,
                       meastime=False):
    """
    Resample an input image to a larger pixel size, conserving the total
    intensity of the image
    """
    if meastime:
        tstart = time.time()

    s = np.array(np.shape(im))

    if newsize is None:
        ns = np.array(np.round(s * oldpfov/newpfov), dtype=int)

        sizerat = newpfov*ns[0] / (oldpfov*s[0])

    else:
        sizerat = 1
        ns = newsize

    if oldpfov is None:
        oldpfov = 1.0

    if newpfov is None:
        newpfov = s[0]/ns[0]


    # --- calculate position grids for the new and old pixels
    xsold = np.arange(s[1]) * oldpfov
    ysold = np.arange(s[0]) * oldpfov

    # --- if the new image is slightly different, offset the grid to be
    #     centered on the old one
    xoff = (1 - sizerat) * oldpfov * s[1] * 0.5
    yoff = (1 - sizerat) * oldpfov * s[0] * 0.5

    xsnew = np.arange(ns[1]) * newpfov + xoff
    ysnew = np.arange(ns[0]) * newpfov + yoff

    newim = np.zeros(ns, dtype=float)

    # --- iterate over then new image, calculate the overlap with the old
    #     pixels and add their intensity to the new pixels acccordingly
#    for y in tqdm(range(ns[0])):
#
#        ysdif = ysnew[y] - ysold
#        ycontr = get_contribution(ysdif, oldpfov, newpfov)
#
#        for x in range(ns[1]):
#
#            xsdif = xsnew[x] - xsold
#            xcontr = get_contribution(xsdif, oldpfov, newpfov)
#
#            newim[y,x] = np.nansum(im * xcontr * ycontr[:, None])
            # print(y, x, newim[y,x], np.nansum(xcontr), np.nansum(ycontr))

    # --- new solution
    ycontr =np.zeros([ns[0],s[0]], dtype=float)
    xcontr =np.zeros([ns[1],s[1]], dtype=float)


    if meastime:
        print(" - INCREASE_PIXELSIZE: everything prepared: ", time.time()-tstart)

    for y in range(ns[0]):
        ysdif = ysnew[y] - ysold
        ycontr[y,:] = get_contribution(ysdif, oldpfov, newpfov)

    for x in range(ns[1]):
        xsdif = xsnew[x] - xsold
        xcontr[x,:] = get_contribution(xsdif, oldpfov, newpfov)


    if meastime:
        print(" - INCREASE_PIXELSIZE: single loops done: ", time.time()-tstart)


    for y in range(ns[0]):
        for x in range(ns[1]):

            # --- identify the contributing pixels
            idx = np.where(xcontr[x,:] > 0)[0]
            idy = np.where(ycontr[y,:] > 0)[0]
#            idx  = np.nonzero(xcontr[x,:] > 0)[0]
#            idy  = np.nonzero(xcontr[y,:] > 0)[0]

            newim[y,x] = np.nansum(im[idy[0]:idy[-1] + 1, idx[0]:idx[-1] + 1]
                                   * xcontr[x, idx] * ycontr[y, idy, None])

#            newim[y,x] = np.nansum(im * xcontr[x, :] * ycontr[y, :, None])

    if meastime:
        print(" - INCREASE_PIXELSIZE: double loop done: ", time.time()-tstart)

    return(newim,sizerat)
