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
from mpl_toolkits.axes_grid1 import ImageGrid


# --- helper function
def chisqg(ydata, ymod, sd=None):

    """
    Returns the chi-square error statistic as the sum of squared errors between
    Ydata(i) and Ymodel(i). If individual standard deviations (array sd) are
    supplied, then the chi-square error statistic is computed as the sum of
    squared errors divided by the standard deviations. Inspired on the IDL
    procedure linfit.pro.
    See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.

    x, y, sd assumed to be Numpy arrays. a, b scalars.
    Returns the float chisq with the chi-square statistic.

    Rodrigo Nemmen
    http://goo.gl/8S1Oo
    """
    # Chi-square statistic (Bevington, eq. 6.9)
    if sd is None:
        chisq = np.sum((ydata - ymod)**2)

    else:
        chisq = np.sum(((ydata - ymod) / sd)**2)

    return chisq



def make_fit_plots(im, fit, outname, xcen, ycen, maxrad, inv=True,
                   cmap='gist_heat', labelcolor='black'):

    """
    make a diagnostic multiplot of a point source fit to an image
    """

    # compute the radial distribution first

    siz = np.shape(im)
    r = np.zeros(siz[0] * siz[1])
    oval = np.zeros(siz[0] * siz[1])
    fval = np.zeros(siz[0] * siz[1])
    z = 0
    for x in range(siz[1]):
        for y in range(siz[0]):
            r[z] = np.sqrt((x - xcen)**2 + (y - ycen)**2)
            oval[z] = im[y, x]
            fval[z] = fit[y, x]
            z = z + 1

    # print(np.min(r), np.max(r), xcen, ycen)

    id = np.argsort(r)
    r = r[id]
    oval = oval[id]
    fval = fval[id]

    chisq = chisqg(oval[r < maxrad], fval[r < maxrad])
    dof = np.sum(r < maxrad)
    red_chisq = chisq / dof

    residual = np.sum(oval[r < maxrad] - fval[r < maxrad])
    # print(chisq, dof, residual)

    # plt.clf()
    # matplotlib.rcdefaults()
    # fig = plt.figure(1, (10, 24))
    fig = plt.figure(figsize=(10, 5.0 / 3.0 * 10 + 2))
    fig.subplots_adjust(bottom=0.2)
    # ax = fig.add_subplot(111)

    if inv is True:
        cmap = cmap+'_r'

    mpl.rc('axes', edgecolor=labelcolor)
    mpl.rc('xtick', color=labelcolor)
    mpl.rc('ytick', color=labelcolor)

    grid = ImageGrid(fig, 211,  # similar to subplot(111)
                     nrows_ncols=(2, 3),  # creates 2x2 grid of axes
                     axes_pad=0.1, aspect=True  # pad between axes in inch.
                     )

    grid[0].imshow(im, cmap=cmap, origin='lower', interpolation='nearest')
    grid[0].set_title('Observation (lin.)', x=0.5, y=0.85, color=labelcolor)

    text = ("Max: " + "{:10.1f}".format(np.max(im))
            + ', STDDEV:' + "{:10.1f}".format(np.std(im)))

    grid[0].text(0.05, 0.1, text, ha='left', va='top',
                 transform=grid[0].transAxes, color=labelcolor, fontsize='8')

    grid[1].imshow(fit, cmap=cmap, origin='lower', interpolation='nearest')
    grid[1].set_title('Fit (lin.)', x=0.5, y=0.85, color=labelcolor)

    grid[2].imshow(im-fit, cmap=cmap, origin='lower', interpolation='nearest')
    grid[2].set_title('Residual (lin.)', x=0.5, y=0.85, color=labelcolor)

    text = ("red. Chi^2: " + "{:10.1f}".format(red_chisq)
            + ', Sum(resid): ' + "{:10.1f}".format(residual))

    grid[2].text(0.05, 0.1, text, ha='left', va='top',
                 transform=grid[2].transAxes, color=labelcolor, fontsize=8)

    grid[3].imshow(im, cmap=cmap, origin='lower', norm=LogNorm(),
                   interpolation='nearest')
    grid[3].set_title('Observation (log.)', x=0.5, y=0.85, color=labelcolor)

    grid[4].imshow(fit, cmap=cmap, origin='lower', norm=LogNorm(),
                   interpolation='nearest')
    grid[4].set_title('Fit (log.)', x=0.5, y=0.85, color=labelcolor)

    grid[5].imshow(im-fit, cmap=cmap, origin='lower', norm=LogNorm(),
                   interpolation='nearest')
    grid[5].set_title('Residual (log.)', x=0.5, y=0.85, color=labelcolor)

    # ----- plots  ----
    # resample for plotting
    samp = 500
    r_s = np.linspace(np.min(r), maxrad, num=samp)
#    r_s = r
#    oval_s = oval
#    fval_s = fval
    oval_s = np.interp(r_s, r[r < maxrad], oval[r < maxrad])
    fval_s = np.interp(r_s, r[r < maxrad], fval[r < maxrad])

    rows = 6
    cols = 3
    s = rows * cols * 0.5

    ax = plt.subplot(rows, cols, s + 1)

    xmin = int(np.round(xcen - maxrad))
    xmax = int(np.round(xcen + maxrad))
    ymin = int(np.round(ycen - maxrad))
    ymax = int(np.round(ycen + maxrad))
    xcen = int(np.round(xcen))
    ycen = int(np.round(ycen))

    plt.plot(im[ycen, xmin:xmax], label='Observation', c='black', linewidth=2)
    plt.plot(fit[ycen, xmin:xmax], label='Fit', c='red')
    plt.title('cut in X (lin)')

    plt.legend(fontsize='6')
    ax.get_xaxis().set_ticklabels([])

    ax = plt.subplot(rows, cols, s + 2)
    plt.plot(im[ymin:ymax, xcen], label='Observation', c='black', linewidth=2)
    plt.plot(fit[ymin:ymax, xcen], label='Fit', c='red')
    plt.title('cut in Y (lin)')
    ax.get_xaxis().set_ticklabels([])

    ax = plt.subplot(rows, cols, s + 3)
    plt.plot(r_s, oval_s, label='Observation', c='black', linewidth=2)
    plt.plot(r_s, fval_s, label='Fit', c='red')
    plt.title('radial (lin)')
    ax.get_xaxis().set_ticklabels([])

    ax = plt.subplot(rows, cols, s + 4)
    plt.plot(np.log10(im[ycen, xmin:xmax]), label='Observation', c='black',
             linewidth=2)
    plt.plot(np.log10(fit[ycen, xmin:xmax]), label='Fit', c='red')
    plt.title('cut in X (log')
    ax.get_xaxis().set_ticklabels([])

    ax = plt.subplot(rows, cols, s + 5)
    plt.plot(np.log10(im[ymin:ymax, xcen]), label='Observation', c='black',
             linewidth=2)
    plt.plot(np.log10(fit[ymin:ymax, xcen]), label='Fit', c='red')
    plt.title('cut in Y (log)')
    ax.get_xaxis().set_ticklabels([])

    ax = plt.subplot(rows, cols, s + 6)
    plt.plot(r_s, np.log10(oval_s), label='Observation', c='black',
             linewidth=2)
    plt.plot(r_s, np.log10(fval_s), label='Fit', c='red')
    plt.title('radial (log)')
    ax.get_xaxis().set_ticklabels([])

    ax = plt.subplot(rows, cols, s + 7)
    plt.plot(im[ycen, xmin:xmax]-fit[ycen, xmin:xmax], label='Residual',
             c='black')
    plt.title('cut in X (lin)')
    plt.legend(fontsize='6')

    ax = plt.subplot(rows, cols, s + 8)
    plt.plot(im[ymin:ymax, xcen]-fit[ymin:ymax, xcen], label='Residual',
             c='black')
    plt.title('cut in Y (lin)')
#    ax.get_xaxis().set_ticklabels([])

    ax = plt.subplot(rows, cols, s + 9)
    plt.plot(r_s, oval_s-fval_s, label='Residual', c='black')
    plt.title('radial (lin)')

    plt.savefig(outname, bbox_inches='tight', pad_inches=0.01)
    plt.clf()

#    matplotlib.rc('axes', edgecolor='black')
#    matplotlib.rc('xtick', color='black')
#    matplotlib.rc('ytick', color='black')
#    matplotlib.rcdefaults()

