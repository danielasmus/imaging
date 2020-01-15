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
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
from matplotlib.patches import Arc
from mpl_toolkits.axes_grid1 import ImageGrid
# from IPython import embed

from astropy.io import fits


from .crop_image import crop_image as _crop_image
from .crosshair_marker import crosshair_marker as _crosshair_marker
from .create_alpha_colmap import create_alpha_colmap as _create_alpha_colmap


# -- helper function
def reshape_input_param(param, nrows, ncols):
    # --- check whether provided parameter is a list or array
    if not isinstance(param, (list, tuple, np.ndarray)):
        newparam = np.full((nrows, ncols), param, dtype=type(param))

    elif len(np.shape(param)) == 1:

        # --- for lists
        if type(param) == list:

            rest = np.shape(param)[0] % nrows

            for i in range((nrows-rest) % nrows):
                param.append(None)

        # --- for numpy arrays
        else:
            param = np.resize(param, nrows*ncols)

        newparam = np.reshape(param, (nrows, ncols))

    else:
        newparam = param

    return(newparam)


#%%

def make_gallery(files=None, ims=None, heads=None, outname=None, pfovs=None,
                 scale="lin", norm=None, absmin=None, absmax=None,
                 permax=None, permin=None, papercol=2, ncols=4, pwidth=None,
                 nrows=None, view_as=None, view_px=None, inv=True,
                 cmap='gist_heat', colorbar=None, cbarwidth=0.03,
                 cbarlabel=None, cbarlabelpad = None, xlabel=None, ylabel=None,
                 xlabelpad=None, ylabelpad=None, titles=None,
                 titcols='black', titx=0.5, tity=0.95, titvalign='top',
                 tithalign='center', subtitles=None, subtitcols='black',
                 subtitx=0.5, subtity=0.05, subtitvalign='bottom',
                 subtithalign='center', plotsym=None, contours=None,
                 alphamap=None, cbartickinterval=None, sbarlength=None,
                 sbarunit='as', sbarcol='black', sbarthick='2',
                 sbarpos=[0.1, 0.05], majtickinterval=None, verbose=False,
                 latex=True, texts=None, textcols='black', textx=0.05,
                 texty=0.95, textvalign='top', texthalign = 'left',
                 textsize=None, replace_NaNs=None, smooth=None, lines=None,
                 latexfontsize=16, axes_pad=0.05
                 ):

    """
    MISSING:
        - implementation to rotate images to North up (and East to the left
          if desired and necessary

    The purpose of this routine is to create puplication-quality multiplots of
    images with a maximum number of customisability. The following parameters
    can be set either for all images a single variable or as an array with the
    same number of elements to set the parameters individually
    INPUT:
        - flles : list of fits files to be plotted
        - pfovs : pixel size in arcsec for the images. If not provided then it
                  will be looked for in the fits headers
        - log : enable logarithmic scaling
        - norm: provide the index of the image that should be normalised to
        - permax: set a percentile value which should be used as the maximum
                  in the colormap
        - papercol: either 1 for a plot fitting into one column in the paper or
                    2 in case the plot is supposed to go over the full page
        - ncols: number of columns of images in the multiplot
        - nrows: number of rows of images in the multiplot
        - view_as: size of the field of view that should be plotted in arcsec
        - inv: if set true then the used colormap is inverted
        - cmap: colormap to be used
        - xlabelpad, xlabelpad : adjust the position of the x,y axis label
        - titles: provide titles for the individual images
        - titcols: colors for the titles
        - titx, tity: x,y position of the title
        - titvalign, tithalign: vertical and horizontal alignment of the title
                                text with respect to the given position
        - subtitles: similar to titles but for another text in the image


    """

    # --- read in the images in a 2D list
    if ims is None:

        if type(files) == list:
            n = len(files)

            if nrows is None and ncols is not None:
                nrows = int(np.ceil(1.0*n/ncols))

            if ncols is None and nrows is not None:
                ncols = int(np.ceil(1.0*n/nrows))

            # --- reshape file list into 2D array
            rest = n % nrows
            for i in range((nrows-rest) % nrows):
                files.append(None)

            files = np.reshape(files, (nrows, ncols))

        elif type(files) == np.ndarray:

            s = np.shape(files)
            if len(s) > 1:
                if nrows is None:
                    nrows = s[0]
                if ncols is None:
                    ncols = s[1]
                n = nrows * ncols

        else:
            n = 1
            nrows = 1
            ncols = 1

            files = np.reshape(files, (-1, ncols))

        # --- create a fill a 2D image array
#        ims = [[None]*ncols]*nrows
#        ims = [None]*nrows*ncols
        ims = [[None]*ncols for _ in range(nrows)]
        #print(np.shape(ims))

        for r in range(nrows):
            for c in range(ncols):
                if files[r][c] != None:
                    if files[r][c] != "":
                        i = ncols * r + c
                        ims[r][c] = fits.getdata(files[r][c],ext=0)

    # images are provides
    else:

        s = np.shape(ims)

        if type(ims) == list or (type(ims) == np.ndarray and len(s) < 4):
            n = len(ims)

            if nrows is None and ncols is not None:
                nrows = int(np.ceil(1.0*n/ncols))

            if ncols is None and nrows is not None:
                ncols = int(np.ceil(1.0*n/nrows))

            # --- reshape file list into 2D array
            rest = n % nrows
            for i in range((nrows-rest) % nrows):
                ims.append(None)

            ims_old = np.copy(ims)
            ims = [[None]*ncols for _ in range(nrows)]

            for r in range(nrows):
                for c in range(ncols):
                    i = ncols * r + c
                    ims[r][c] = ims_old[i]

        elif type(ims) == np.ndarray:


            if len(s) > 1:
                if nrows is None:
                    nrows = s[0]
                if ncols is None:
                    ncols = s[1]
                n = nrows * ncols

        else:
            n = 1
            nrows = 1
            ncols = 1

            ims = [[ims]]



    if heads is None and files is not None:
        heads =  [[None]*ncols for _ in range(nrows)]
        for r in range(nrows):
            for c in range(ncols):
                if files[r][c] != None:
                    if files[r][c] != "":
                        heads[r][c] = fits.getheader(files[r][c],ext=0)
#    print(np.shape(heads), len(np.shape(heads)))
    if heads is not None:
        if len(np.shape(heads)) == 1:
            heads = np.full((nrows, ncols), heads, dtype=object)

    if pfovs is None and heads is not None:
        pfovs = [[None]*ncols for _ in range(nrows)]
        for r in range(nrows):
            for c in range(ncols):
                if heads[r][c] != None:
                    if heads[r][c] != "":
                        if "PFOV" in heads[r][c]:
                            pfovs[r][c] = float(heads[r][c]['PFOV'])
                        elif "HIERARCH ESO INS PFOV" in heads[r][c]:
                            pfovs[r][c] = float(heads[r][c]["HIERARCH ESO INS PFOV"])
                        elif "CDELT1" in heads[r][c]:
                            pfovs[r][c] = np.abs(heads[r][c]["CDELT1"])*3600
        if len(pfovs) == 0:
            pfovs = None

    else:
        pfovs = reshape_input_param(pfovs, nrows, ncols)

    scale = reshape_input_param(scale, nrows, ncols)
    permin = reshape_input_param(permin, nrows, ncols)
    permax = reshape_input_param(permax, nrows, ncols)
    absmin = reshape_input_param(absmin, nrows, ncols)
    absmax = reshape_input_param(absmax, nrows, ncols)
    norm = reshape_input_param(norm, nrows, ncols)
    smooth = reshape_input_param(smooth, nrows, ncols)
    titles = reshape_input_param(titles, nrows, ncols)
    titcols = reshape_input_param(titcols, nrows, ncols)
    texts = reshape_input_param(texts, nrows, ncols)
    textcols = reshape_input_param(textcols, nrows, ncols)
    subtitles = reshape_input_param(subtitles, nrows, ncols)
    subtitcols = reshape_input_param(subtitcols, nrows, ncols)
    sbarlength = reshape_input_param(sbarlength, nrows, ncols)

    if lines is not None:
        lines = reshape_input_param(lines, nrows, ncols)

    if verbose:

#        ny, nx = ims[0].shape
        print("MAKE_GALLERY: nrows, ncols: " , nrows, ncols)
        print("MAKE_GALLERY: n: ", n)
        print("MAKE_GALLERY: pfovs: ", pfovs)
        print("MAKE_GALLERY: scales: ", scale)
 #       print("MAKE_GALLERY: nx, ny: ", nx, ny)


    # --- set up the plotting configuration to use latex


    if latex:
        mpl.rcdefaults()


        mpl.rc('font',**{'family':'sans-serif', 'serif':['Computer Modern Serif'],
                         'sans-serif':['Helvetica'], 'size':latexfontsize,
                        'weight':500, 'variant':'normal'})

        mpl.rc('axes',**{'labelweight':'normal', 'linewidth':1.5})
        mpl.rc('ytick',**{'major.pad':8, 'color':'k'})
        mpl.rc('xtick',**{'major.pad':8})
        mpl.rc('mathtext',**{'default':'regular','fontset':'cm',
                             'bf':'monospace:bold'})

        mpl.rc('text', **{'usetex':True})
        mpl.rc('text.latex',preamble=r'\usepackage{cmbright},\usepackage{relsize},'+\
                                    r'\usepackage{upgreek}, \usepackage{amsmath}'+\
                                    r'\usepackage{bm}')

        mpl.rc('contour', **{'negative_linestyle':'solid'}) # dashed | solid


    plt.clf()


    # 14.17
    # 6.93
    if pwidth == None:
        if papercol == 1:
            pwidth = 6.93
        elif papercol == 2:
            pwidth = 14.17
        else:
            pwidth = 7 * papercol

    subpwidth = pwidth/ncols


    if verbose:
         print("MAKE_GALLERY: pwidth, subpwidth: " , pwidth, subpwidth)

    fig = plt.figure(figsize=(pwidth, nrows*subpwidth))
    # fig.subplots_adjust(bottom=0.2)
    # fig.subplots_adjust(left=0.2)

    if inv:
        cmap = cmap+'_r'

    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols),
                     axes_pad=axes_pad,
                     aspect=True)

    handles = []
    ahandles = []
    vmin = np.zeros((nrows, ncols))
    vmax = np.zeros((nrows, ncols))
    amin = np.zeros((nrows, ncols))
    amax = np.zeros((nrows, ncols))

    # --- main plotting loop
    for r in range(nrows):
        for c in range(ncols):

            i = ncols * r + c

            if verbose:
                print("MAKE_GALLERY: r,c,i" , r,c,i)
                print(" - smooth: ", smooth[r][c])

            if np.shape(ims[r][c]) == ():
                continue

            if view_as is not None:
                view_px = view_as / pfovs[r][c]
                im = _crop_image(ims[r][c], box=np.ceil(view_px/2.0)*2+2)
            else:
                im = np.copy(ims[r][c])

            #print(r,c, np.nanmax(ims[r][c]), np.nanmax(im))


            if smooth[r][c]:
                im = gaussian_filter(im, sigma=smooth[r][c], mode='nearest')

            if norm[r][c]:
                im[0,0] = np.nanmax(ims[int(norm[r][c])])

            # --- adjust the cut levels
            if permin[r][c]:
                vmin[r][c] = np.nanpercentile(im, float(permin[r][c]))
            else:
                vmin[r][c] = np.nanmin(im)

            if permax[r][c]:
                vmax[r][c] = np.nanpercentile(im, float(permax[r][c]))
            else:
                vmax[r][c] = np.nanmax(im)

            if absmax[r][c] is not None:
                vmax[r][c] = absmax[r][c]

            if absmin[r][c] is not None:
                vmin[r][c] = absmin[r][c]


            sim = np.copy(im)
            sim[im < vmin[r][c]] = vmin[r][c]
            sim[im > vmax[r][c]] = vmax[r][c]

            if replace_NaNs is not None:
                idn = np.nonzero(np.isnan(sim))
    #            print(i, len(idn), idn[0])
                if len(idn) == 0:
                    continue
                elif replace_NaNs == "min":
                    sim[np.ix_(idn[0], idn[1])] = vmin[r][c]
                elif replace_NaNs == "max":
                    sim[np.ix_(idn[0], idn[1])] = vmax[r][c]
                else:
                    sim[np.ix_(idn[0], idn[1])] = replace_NaNs


            # --- logarithmic scaling?
            if scale[r][c] == "log":
                sim = np.log10(1000.0 * (sim - vmin[r][c]) /
                              (vmax[r][c] - vmin[r][c]) + 1)



            if verbose:
                print("MAKE_GALLERY: scale[r][c]" , scale[r][c])
                print("MAKE_GALLERY: vmin[r][c], vmax[r][c]" , vmin[r][c], vmax[r][c])


            handle = grid[i].imshow(sim, cmap=cmap, origin='lower',
                                     interpolation='nearest')


            ny, nx = im.shape

            if verbose:
                print("ny, nx:", ny,nx)

            if pfovs is not None:

                xmin =  (nx/2 -1) * pfovs[r][c]
                xmax =  -nx/2 * pfovs[r][c]
                ymin = -ny/2 * pfovs[r][c]
                ymax = (ny/2 -1) * pfovs[r][c]
                if xlabel is None:
                    xlabel = 'RA offset ["]'
                if ylabel is None:
                    ylabel = 'DEC offset ["]'

            else:

                xmax =  (nx/2 -1)
                xmin =  -nx/2
                ymin = -ny/2
                ymax = (ny/2 -1)
                if xlabel is None:
                    xlabel = 'x offset [px]'
                if ylabel is None:
                    ylabel = 'y offset [px]'

            # print(pfovs[r][c])
            # pdb.set_trace()
            # set the extent of the image
            extent = [xmin, xmax, ymin, ymax]

            if verbose:
                print("extent:", extent)
                print("x/ylabel:", xlabel, ylabel)

            handle.set_extent(extent)



            # --- optionally overplot a second map using transparency
            if alphamap is not None:

                uim = np.copy(alphamap[r][c]['im'])

                # --- determine the levels
                if alphamap[r][c]['min'] is None:
                    amin[r][c] = np.nanmin(im)
                elif str(alphamap[r][c]['min']) == 'vmax':
                    amin[r][c] = vmax[r][c]
                elif str(alphamap[r][c]['min']) == 'vmin':
                    amin[r][c] = vmin[r][c]
                else:
                    amin[r][c] = np.nanpercentile(im, float(alphamap[r][c]['min']))

                if alphamap[r][c]['max'] is None:
                    amax[r][c] = np.nanmax(im)
                elif str(alphamap[r][c]['max']) == 'vmax':
                    amax[r][c] = vmax[r][c]
                elif str(alphamap[r][c]['max']) == 'vmax':
                    amax[r][c] = vmin[r][c]
                else:
                    amax[r][c] = np.nanpercentile(im, float(alphamap[r][c]['max']))

                uim[uim < amin[r][c]] = amin[r][c]
                uim[uim > amax[r][c]] = amax[r][c]

                cmap2 = _create_alpha_colmap(alphamap[r][c]['mincolor'],
                                            alphamap[r][c]['maxcolor'],
                                            alphamap[r][c]['minalpha'],
                                            alphamap[r][c]['maxalpha'])

                if alphamap[r][c]['log']:
                    unorm = LogNorm()
    #                uim = np.log10(1000.0 * (uim - amin[r][c]) /
    #                          (amax[r][c] - amin[r][c]) + 1)

                else:
                    unorm = None

                ahandle = grid[i].imshow(uim, cmap=cmap2, origin='lower',
                                         interpolation='nearest', norm=unorm)

                ahandle.set_extent(extent)
                ahandles.append(ahandle)


            # --- optionally draw contours
            if contours is not None:

                # --- determine the levels
                if contours[r][c]['min'] is None:
                    cmin = np.nanmin(im)
                elif str(contours[r][c]['min']) == 'vmax':
                    cmin = vmax[r][c]
                elif str(contours[r][c]['min']) == 'vmin':
                    cmin = vmin[r][c]
                else:
                    cmin = contours[r][c]['min']

                if contours[r][c]['max'] is None:
                    cmax = np.nanmax(im)
                elif str(contours[r][c]['max']) == 'vmax':
                    cmax = vmax[r][c]
                elif str(contours[r][c]['max']) == 'vmax':
                    cmax = vmin[r][c]
                else:
                    cmax = contours[r][c]['max']

                if contours[r][c]['nstep'] is None:
                    nstep = 10
                else:
                    nstep = contours[r][c]['nstep']

                if contours[r][c]['stepsize'] is None:
                    stepsize = int(cmax-cmin/nstep)
                else:
                    stepsize = contours[r][c]['stepsize']

                levels = np.arange(cmin, cmax, stepsize)

                cont = grid[i].contour(im, levels=levels, origin='lower',
                                       colors=contours[r][c]['color'],
                                       linewidth=contours[r][c]['linewidth'],
                                       extent=extent)

                grid[i].clabel(cont, levels[1::contours[r][c]['labelinterval']], inline=1,
                               fontsize=contours[r][c]['labelsize'],
                               fmt=contours[r][c]['labelfmt'])



            if pfovs is not None and view_as is not None:
                xmin = 0.5*view_as
                xmax = -0.5*view_as
                ymin = -0.5*view_as
                ymax = 0.5*view_as

            elif view_px is not None:
                xmin = -0.5*view_px
                xmax = 0.5*view_px
                ymin = -0.5*view_px
                ymax = 0.5*view_px

            if verbose:
                print("xmin,xmax,ymin,ymax:", xmin,xmax,ymin,ymax)

            grid[i].set_ylim(ymin, ymax)
            grid[i].set_xlim(xmin, xmax)



            # --- optionally draw some lines
            if lines is not None:

                nlin = len(lines[r][c]["length"])
                for l in range(nlin):

                    # --- provided unit for lines is arcsec?
    #                print(lines[r][c]["length"][l])
                    if lines[r][c]["unit"][l] == "px":
                        lines[r][c]["length"][l] *= pfovs[r][c]
                        lines[r][c]["xoff"][l] *= pfovs[r][c]
                        lines[r][c]["yoff"][l] *= pfovs[r][c]

    #                print(lines[r][c]["length"][l])

                    ang_rad = (90 - lines[r][c]["pa"][l])  * np.pi/180.0
                    hlength = 0.5*lines[r][c]["length"][l]

                    lx = [lines[r][c]["xoff"][l] - hlength*np.cos(ang_rad),
                          lines[r][c]["xoff"][l] + hlength*np.cos(ang_rad)]

                    ly = [lines[r][c]["yoff"][l] - hlength*np.sin(ang_rad),
                          lines[r][c]["yoff"][l] + hlength*np.sin(ang_rad)]

                    grid[i].plot(lx, ly, color=lines[r][c]["color"][l],
                                 linestyle=lines[r][c]["style"][l], marker='',
                                 linewidth=lines[r][c]["thick"][l],
                                 alpha=lines[r][c]["alpha"][l])

    #                theta1 = lines[r][c]["pa"][l] - 10
    #                theta2 = lines[r][c]["pa"][l] + 10

                    # --- angular error bars
                    if lines[r][c]["pa_err"][l] > 0:

                        theta1 = 90 - lines[r][c]["pa"][l] - lines[r][c]["pa_err"][l]
                        theta2 = 90 - lines[r][c]["pa"][l] + lines[r][c]["pa_err"][l]

                        parc = Arc((lines[r][c]["xoff"][l], lines[r][c]["yoff"][l]),
                                   width=2*hlength, height=2*hlength,
                                   color=lines[r][c]["color"][l],
                                   linewidth=lines[r][c]["thick"][l],
                                   angle=0, theta1=theta1, theta2=theta2,
                                   alpha=lines[r][c]["alpha"][l])

                        grid[i].add_patch(parc)


                        theta1 = -90 - lines[r][c]["pa"][l] - lines[r][c]["pa_err"][l]
                        theta2 = -90 - lines[r][c]["pa"][l] + lines[r][c]["pa_err"][l]

                        parc = Arc((lines[r][c]["xoff"][l], lines[r][c]["yoff"][l]),
                                   width=2*hlength, height=2*hlength,
                                   color=lines[r][c]["color"][l],
                                   linewidth=lines[r][c]["thick"][l],
                                   angle=0, theta1=theta1, theta2=theta2,
                                   alpha=lines[r][c]["alpha"][l])

                        grid[i].add_patch(parc)

            # --- optionally draw some symbols
            if plotsym is not None:
       #             if hasattr(plotsym[r][c]['x'], "__len__"):
                nsymplots = len(plotsym[r][c]['x'])
                print('nsymplots: ', nsymplots)
                for j in range(nsymplots):
    #                grid[i].plot(plotsym[r][c]['x'][j], plotsym[r][c]['y'][j],
    #                             marker=plotsym[r][c]['marker'][j],
    #                             markersize=plotsym[r][c]['markersize'][j],
    #                             markeredgewidth=plotsym[r][c]['markeredgewidth'][j],
    #                             fillstyle=plotsym[r][c]['fillstyle'][j],
    #                             alpha=plotsym[r][c]['alpha'][j],
    #                             markerfacecolor=plotsym[r][c]['markerfacecolor'][j],
    #                             markeredgecolor=plotsym[r][c]['markeredgecolor'][j])

                    if plotsym[r][c]['marker'][j] == 'ch':
                        marker = _crosshair_marker()
                    elif plotsym[r][c]['marker'][j] == 'ch45':
                        marker = _crosshair_marker(pa=45)
                    else:
                        marker = plotsym[r][c]['marker'][j]

                    grid[i].scatter(plotsym[r][c]['x'][j], plotsym[r][c]['y'][j],
                                 marker=marker,
                                 s=plotsym[r][c]['size'][j],
                                 linewidths=plotsym[r][c]['linewidth'][j],
                                 alpha=plotsym[r][c]['alpha'][j],
                                 facecolor=plotsym[r][c]['color'][j],
                                 edgecolors=plotsym[r][c]['edgecolor'][j],
                                 linestyle=plotsym[r][c]['linestyle'][j])

                    if plotsym[r][c]['label'][j] is not None:
                        grid[i].text(plotsym[r][c]['x'][j][0]+plotsym[r][c]['labelxoffset'][j],
                                     plotsym[r][c]['y'][j][0]+plotsym[r][c]['labelyoffset'][j],
                                     plotsym[r][c]['label'][j],
                                     color=plotsym[r][c]['labelcolor'][j],
                                     verticalalignment=plotsym[r][c]['labelvalign'][j],
                                     horizontalalignment=plotsym[r][c]['labelhalign'][j])

            # --- ticks
            if majtickinterval is None:

                majorLocator = MultipleLocator(1)

                if ymax - ymin < 2:
                    majorLocator = MultipleLocator(0.5)
                    minorLocator = AutoMinorLocator(5)

                elif (ymax - ymin > 10) & (ymax - ymin <= 20):
                    majorLocator = MultipleLocator(2)

                elif (ymax - ymin > 20) & (ymax - ymin <= 100):
                    majorLocator = MultipleLocator(5)

                elif ymax - ymin > 100:
                    majorLocator = MultipleLocator(10)


            else:
                majorLocator = MultipleLocator(majtickinterval)

            minorLocator = AutoMinorLocator(10)
            if ymax - ymin < 2:
                minorLocator = AutoMinorLocator(5)

            grid[i].xaxis.set_minor_locator(minorLocator)
            grid[i].xaxis.set_major_locator(majorLocator)
            grid[i].yaxis.set_minor_locator(minorLocator)
            grid[i].yaxis.set_major_locator(majorLocator)
            grid[i].yaxis.set_tick_params(width=1.5, which='both')
            grid[i].xaxis.set_tick_params(width=1.5, which='both')
            grid[i].xaxis.set_tick_params(length=6)
            grid[i].yaxis.set_tick_params(length=6)
            grid[i].xaxis.set_tick_params(length=3, which='minor')
            grid[i].yaxis.set_tick_params(length=3, which='minor')




            # --- text
            if titles[r][c]:
              #   pdb.set_trace()

                if verbose:
                    print("titles[r][c]", titles[r][c])

                grid[i].set_title(titles[r][c], x=titx, y=tity, color=titcols[r][c],
                                  verticalalignment=titvalign,
                                  horizontalalignment=tithalign)

            if subtitles[r][c]:

                if verbose:
                    print("subtitles[r][c]", subtitles[r][c])

                grid[i].text(subtitx, subtity, subtitles[r][c], color=subtitcols[r][c],
                             transform=grid[i].transAxes,
                             verticalalignment=subtitvalign,
                             horizontalalignment=subtithalign)

            if texts[r][c]:
                if verbose:
                    print("texts[r][c]", texts[r][c])
                grid[i].text(textx, texty, texts[r][c], color=textcols[r][c],
                             transform=grid[i].transAxes,
                             verticalalignment=textvalign,
                             horizontalalignment=texthalign, fontsize=textsize)


            # --- scale bar for size comparison
            if sbarlength[r][c]:
                if sbarunit == 'px':
                    sbarlength[r][c] = sbarlength[r][c] * pfovs[r][c]
                sx = [sbarpos[0]- 0.5*sbarlength[r][c], sbarpos[0]+ 0.5*sbarlength[r][c]]
                sy = [sbarpos[1], sbarpos[1]]
                grid[i].plot(sx, sy, linewidth=sbarthick, color=sbarcol,
                        transform=grid[i].transAxes)

            handles.append(handle)

    # --- get the extent of the largest box containing all the axes/subplots
    extents = np.array([bla.get_position().extents for bla in grid])
    bigextents = np.empty(4)
    bigextents[:2] = extents[:, :2].min(axis=0)
    bigextents[2:] = extents[:, 2:].max(axis=0)

    # --- distance between the external axis and the text
    if xlabelpad is None and papercol == 2:
        xlabelpad = 0.03
    elif xlabelpad is None and papercol == 1:
        xlabelpad = 0.15
    elif xlabelpad is None:
        xlabelpad = 0.15 / papercol * 0.5

    if ylabelpad is None and papercol == 2:
        ylabelpad = 0.055
    elif ylabelpad is None and papercol == 1:
        ylabelpad = 0.1
    elif ylabelpad is None:
        ylabelpad = 0.1 / papercol

    if verbose:
        print("xlabelpad,ylabelpad: ", xlabelpad,ylabelpad)

    # --- text to mimic the x and y label. The text is positioned in
    #     the middle
    fig.text((bigextents[2] + bigextents[0])/2,
             bigextents[1] - xlabelpad, xlabel,
             horizontalalignment='center', verticalalignment='bottom')

    fig.text(bigextents[0] - ylabelpad,
             (bigextents[3] + bigextents[1])/2, ylabel,
             rotation='vertical', horizontalalignment='left',
             verticalalignment='center')


    # --- now colorbar business:
    if colorbar is not None:
        # first draw the figure, such that the axes are positionned
        fig.canvas.draw()
        #create new axes according to coordinates of image plot
        trans = fig.transFigure.inverted()


    if colorbar == 'row':
        for i in range(nrows):
            g = grid[i*ncols - 1].bbox.transformed(trans)

            if alphamap[ncols, i] is not None:

                height = 0.5*g.height
                pos = [g.x1 + g.width*0.02, g.y0+height, g.width*cbarwidth,
                       height]
                cax = fig.add_axes(pos)

                if alphamap[ncols,i]['log']:
                    formatter = LogFormatter(10, labelOnlyBase=False)
                    cb = plt.colorbar(ticks=[0.1,0.2,0.5,1,5,10,20,50,100,200,500,1000,2000,5000,10000],
                                      format=formatter,
                                      mappable=ahandles[i*ncols - 1], cax=cax)
                else:
                    cb = plt.colorbar(mappable=ahandles[i*ncols - 1], cax=cax)

                #majorLocator = MultipleLocator(alphamap[i*ncols - 1]['cbartickinterval'])
                #cb.ax.yaxis.set_major_locator(majorLocator)

#                if alphamap[i*ncols - 1]['log']:
#                    oldlabels = cb.ax.get_yticklabels()
#                    oldlabels = np.array([float(x.get_text().replace('$','')) for x in oldlabels])
#                    newlabels = 0.001*(10.0**oldlabels -1)*(amax[i*ncols - 1] - amin[i*ncols - 1]) + amin[i*ncols - 1]
#                    newlabels = [str(x)[:4] for x in newlabels]
#                    cb.ax.set_yticklabels(newlabels)


            else:

                height = g.height

            # --- pos = [left, bottom, width, height]
            pos = [g.x1 + g.width*0.02, g.y0, g.width*cbarwidth, height ]
            cax = fig.add_axes(pos)

            if (vmax[ncols,i] - vmin[ncols,i]) < 10:
                decimals = 1
            else:
                decimals = 0
            if (vmax[ncols,i] - vmin[ncols,i]) < 1:
                decimals = 2

            ticks = np.arange(vmin[ncols,i]+1.0/10**decimals, vmax[ncols,i], cbartickinterval)

            ticks = np.round(ticks, decimals=decimals)

            cb = plt.colorbar(ticks=ticks, mappable=handles[i*ncols - 1], cax=cax)

            # majorLocator = MultipleLocator(cbartickinterval)
            # cb.ax.yaxis.set_major_locator(majorLocator)

            if scale[ncols,i] == "log":
                oldlabels = cb.ax.get_yticklabels()
                oldlabels = np.array([float(x.get_text().replace('$','')) for x in oldlabels])
                newlabels = 0.001*(10.0**oldlabels -1)*(vmax[i*ncols - 1] - vmin[ncols,i]) + vmin[ncols,i]
                newlabels = [str(x)[:4] for x in newlabels]
                cb.ax.set_yticklabels(newlabels)

    elif colorbar == 'single':

            gb = grid[-1].bbox.transformed(trans)
            gt = grid[0].bbox.transformed(trans)

            if alphamap is not None:
                height = 0.5*(gt.y1-gb.y0)
                pos = [gb.x1 + gb.width*0.02, gb.y0+height, gb.width*cbarwidth,
                       height]
                cax = fig.add_axes(pos)
                # cb = plt.colorbar(mappable=ahandles[0], cax=cax)

 #               majorLocator = MultipleLocator(alphamap[0]['cbartickinterval'])
 #               cb.ax.yaxis.set_major_locator(majorLocator)

#                if alphamap[0]['log']:
#                   oldlabels = cb.ax.get_yticklabels()
#                   oldlabels = np.array([float(x.get_text().replace('$','')) for x in oldlabels])
#                   newlabels = 0.001*(10.0**oldlabels -1)*(amax[0] - amin[0]) + amin[0]
#                   newlabels = [str(x)[:4] for x in newlabels]
#                   cb.ax.set_yticklabels(newlabels)
                if alphamap[0]['log']:
                    formatter = LogFormatter(10, labelOnlyBase=False)
                    cb = plt.colorbar(ticks=[0.1,0.2,0.5,1,5,10,20,50,100,200,500,1000,2000,5000,10000],
                                      format=formatter,
                                      mappable=ahandles[0], cax=cax)
                else:
                    cb = plt.colorbar(mappable=ahandles[0], cax=cax)


            else:
                height = (gt.y1-gb.y0)

            # --- pos = [left, bottom, width, height]
            pos = [gb.x1 + gb.width*0.02, gb.y0, gb.width*cbarwidth, height]
            cax = fig.add_axes(pos)
            cb = plt.colorbar(mappable=handles[0], cax=cax)

            # majorLocator = MultipleLocator(cbartickinterval)
            # cb.ax.yaxis.set_major_locator(majorLocator)

            if scale[0,0] == "log":
                oldlabels = cb.ax.get_yticklabels()
                oldlabels = np.array([float(x.get_text().replace('$','')) for x in oldlabels])
                newlabels = 0.001*(10.0**oldlabels -1)*(vmax[0] - vmin[0]) + vmin[0]
                newlabels = [str(x)[:4] for x in newlabels]
                cb.ax.set_yticklabels(newlabels)

    if cbarlabel is not None:

        if cbarlabelpad is None:
            if papercol == 1:
                cbarlabelpad = 0.15
            else:
                cbarlabelpad = 0.08

        fig.text(bigextents[2] + cbarlabelpad,
             (bigextents[3] + bigextents[1])/2, cbarlabel,
             rotation='vertical', horizontalalignment='right',
             verticalalignment='center')


    if outname:
        plt.savefig(outname, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    else:
        plt.show()


    if latex:
        mpl.rcdefaults()
