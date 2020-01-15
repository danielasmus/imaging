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
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Column

from .get_pointsource import get_pointsource as _get_pointsource


def make_ext_measurements(obsfits, step=0.4, nstep=11, posang=0, fout=None,
                          show_plots=False):

    """
    take a number of input fits images and measure the flux in consecutive
    apertures along two rows through the central source, one horizontal and one
    vertical. Optionally, the images can be rotated before by providing a
    position angle
    """

    if fout is not None:
        out = open(fout, 'w')

        out.write('x,y\n')

        for x in range(nstep):
            out.write(str(step*(x+0.5*(1-nstep)))+', 0\n')

        for y in range(nstep):
            out.write('0, '+str(step*(y+0.5*(1-nstep)))+'\n')

        out.close()


    for j in range(len(obsfits)):

        im = fits.getdata(obsfits[j])
        head = fits.getheader(obsfits[j])

        #plt.imshow(im, origin='bottom', interpolation='nearest', norm=LogNorm())

        pfov = head['PFOV']
        filt = head['Filter']
        box = step/pfov

        print(filt,pfov)

        # --- first rotate the image such that the bar is horizontal
        rotim = ndimage.interpolation.rotate(im, posang, order=3)

        if show_plots is  True:
            plt.imshow(rotim, origin='bottom', interpolation='nearest', norm=LogNorm())
            plt.show()

        # --- redetermine the center
        params, _, _ = _get_pointsource(rotim, method='mpfit')

        xcen = params[2]
        ycen = params[3]

        ymin = ycen - 0.5*box
        ymax = ymin + box

        fluxes = np.zeros(2*nstep)

        s = np.shape(rotim)

        for i in range(nstep):
# for i in range(1):

            xmin = xcen - 0.5 * nstep * box + i * box
            xmax = xmin + box

            # --- make a contr mask
            c = np.zeros(s)

            for x in range(s[1]):
                for y in range(s[0]):
                    c[y,x] = (np.max([0,np.min([1,x-xmin])])
                             *np.max([0,np.min([1,xmax-x])])
                             *np.max([0,np.min([1,y-ymin])])
                             *np.max([0,np.min([1,ymax-y])]))

            cim = np.multiply(rotim,c)
            if show_plots is  True:
                plt.imshow(cim,origin='bottom', interpolation='nearest')
                plt.show()

            fluxes[i] = np.sum(cim)

    #fluxes[i] = np.sum(rotim[int(np.ceil(ymin)):int(np.floor(ymax)),
    #                         int(np.ceil(xmin)):int(np.floor(xmax))])

            print(xmin,ymin,fluxes[i])

        xmin = xcen - 0.5*box
        xmax = xmin + box

        for i in range(nstep):
# for i in range(1):

            ymin = ycen - 0.5 * nstep * box + i * box
            ymax = ymin + box

            # --- make a contr mask
            c = np.zeros(s)

            for x in range(s[1]):
                for y in range(s[0]):
                    c[y,x] = (np.max([0,np.min([1,x-xmin])])
                             *np.max([0,np.min([1,xmax-x])])
                             *np.max([0,np.min([1,y-ymin])])
                             *np.max([0,np.min([1,ymax-y])]))

            cim = np.multiply(rotim,c)
            if show_plots is  True:
                plt.imshow(cim,origin='bottom', interpolation='nearest')
                plt.show()

            fluxes[i+nstep] = np.sum(cim)

        #fluxes[i] = np.sum(rotim[int(np.ceil(ymin)):int(np.floor(ymax)),
        #                         int(np.ceil(xmin)):int(np.floor(xmax))])

            print(xmin,ymin,fluxes[i+nstep])

            # ---- print the data into file
            d = ascii.read(fout, header_start=0, delimiter=',', guess=False)

        newCol = Column(fluxes, name=filt)
        d.add_column(newCol)

        d.write(fout, delimiter=',', format='ascii',
                fill_values=[(ascii.masked, '')])

    return(d)

