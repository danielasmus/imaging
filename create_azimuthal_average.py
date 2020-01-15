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
from matplotlib.colors import LogNorm
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
from astropy.io import fits

from .get_pointsource import get_pointsource as _get_pointsource



def create_azimuthal_average(im=None, infits=None, head=None, step=0.5,
                             outfits=None, show_plots=True):

    """
    Create an azimuthally averaged (actually median) image from a given image.
    For unresolved point sources this is a good representation of the PSF
    """

    if im is None:
        im = fits.getdata(infits)

    if head is None and outfits is not None:
        head = fits.getheader(infits)


    # --- find the center of the image
    params, _, _ = _get_pointsource(im, method='mpfit')

    xcen = params[2]
    ycen = params[3]
    s = np.shape(im)

    # --- compute the radial distance from the center
    rdist = np.zeros(s)

    for x in range(s[1]):
        for y in range(s[0]):
            rdist[y,x] = np.sqrt((x - xcen)**2 + (y - ycen)**2)


    #plt.imshow(rdist, origin='bottom', interpolation='nearest')

    # --- deform 2D arrays into 1D
    im1D = im.flatten()
    rdist1D = rdist.flatten()

    id = np.argsort(rdist1D)
    rdist1D = rdist1D[id]
    im1D = im1D[id]

    #med = scipy.signal.medfilt(im1D,kernel_size=11)

    rmax = np.max(rdist1D)
    nstep = int(np.ceil(rmax/step))
    med = np.zeros(nstep)
    medr = np.zeros(nstep)

    for r in range(nstep):

        rmin = r * step
        rmax = rmin + step

        id = (rdist1D > rmin) & (rdist1D <= rmax)
        med[r] = np.median(im1D[id])
        medr[r] = rmin + 0.5 * step
        # print(r,medr[r],med[r], np.sum(id))

    if show_plots is True:
        plt.clf()
        plt.plot(rdist1D,np.log10(im1D))
        plt.xlim(0,50)
        plt.plot(medr,np.log10(med), color='red')
        plt.show()

    # ---- clean the array
    id = ~np.isnan(med)
    med = med[id]
    medr = medr[id]
    nstep = len(med)

    azim = np.zeros(s)

    for x in range(s[1]):
        for y in range(s[0]):

       # id = np.where((rdist[y,x] > medr - 0.5*step) & (rdist[y,x] < medr + 0.5*step))[0]
       # print(x,y,id)
       # cim[y,x] = im[y,x] - med[id]
       # sim[y,x] =  med[id]

            overlap = np.zeros(nstep)

            # --- all rings that are fully within the pixel
            id = np.where((medr + 0.5*step <= rdist[y,x]+0.5)
                          & (medr - 0.5*step >= rdist[y,x]-0.5))[0]
            if len(id) > 0:
                overlap[id] = 1

            # --- the ring at the lower end
            id = np.where((medr + 0.5*step > rdist[y,x]-0.5)
                          & (medr - 0.5*step < rdist[y,x]-0.5))[0]
            if len(id) > 0:
                overlap[id] = medr[id] + 0.5*step - (rdist[y,x] - 0.5)

            # --- the ring at the higher end
            id = np.where((medr + 0.5*step > rdist[y,x]+0.5)
                          & (medr - 0.5*step < rdist[y,x]+0.5))[0]
            if len(id) > 0:
                overlap[id] = (rdist[y,x] + 0.5) - (medr[id] - 0.5*step)

            azim[y,x] = np.average(med, weights=overlap)

    if show_plots is True:
        plt.clf()
        plt.imshow(azim, origin='bottom', interpolation='nearest', norm=LogNorm())
        plt.show()

    if outfits:
        fits.writeto(outfits, azim, head, overwrite=True)

    return(azim)



