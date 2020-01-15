#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-15: created by Daniel Asmus


NOTES:
    - specific set of routines to simulate 'observations' of SKIRT model cubes
      (used for the case of Circinus with VISIR; Stalevski et al. 2017)

TO-DO:
    -
"""


import numpy as np
import time
from scipy import ndimage
import scipy.signal
# import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
from mpl_toolkits.axes_grid1 import ImageGrid
# from IPython import embed

from astropy.io import fits
from astropy.io import ascii

from astropy.modeling import models, fitting
from astropy.modeling import Fittable2DModel, Parameter

from .crop_image import crop_image as _crop_image
from .get_pointsource import get_pointsource as _get_pointsource
from .increase_pixelsize import increase_pixelsize as _increase_pixelsize
from .extend_image import extend_image as _extend_image
from .make_fit_plots import make_fit_plots as _make_fit_plots
from .simple_image_plot import simple_image_plot as _simple_image_plot


# s2f = (2.0 * np.sqrt(2.0*np.log(2.0)))
# f2s = 1.0/s2f

#%%

# --- class for obscured Airy Disk
class obsc_AiryDisk2D(Fittable2DModel):

    amplitude = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    radius = Parameter(default=1)
    obsrat = Parameter(default=0.15)
    _j1 = None

    def __init__(self, amplitude=amplitude.default, x_0=x_0.default,
                 y_0=y_0.default, radius=radius.default,
                 obsrat=obsrat.default, **kwargs):

        if self._j1 is None:
            try:
                from scipy.special import j1, jn_zeros
                self.__class__._j1 = j1
                self.__class__._rz = jn_zeros(1, 1)[0] / np.pi
            # add a ValueError here for python3 + scipy < 0.12
            except ValueError:
                raise ImportError("AiryDisk2D model requires scipy > 0.11.")

        super(obsc_AiryDisk2D, self).__init__(
            amplitude=amplitude, x_0=x_0, y_0=y_0, radius=radius,
            obsrat=obsrat, **kwargs)

    # TODO: Why does this particular model have its own special __deepcopy__
    # and __copy__?  If it has anything to do with the use of the j_1 function
    # that should be reworked.
#    def __deepcopy__(self, memo):
#        new_model = self.__class__(self.amplitude.value, self.x_0.value,
#                                   self.y_0.value, self.radius.value,
#                                   self.obsrat.value)
#        return new_model
#
#    def __copy__(self):
#        new_model = self.__class__(self.amplitude.value, self.x_0.value,
#                                   self.y_0.value, self.radius.value,
#                                   self.obsrat.value)
#        return new_model

    @classmethod
    def evaluate(cls, x, y, amplitude, x_0, y_0, radius, obsrat):
        """Two dimensional Airy model function with central obscuration"""

        r = np.sqrt((x - x_0)**2 + (y - y_0)**2) / (radius / cls._rz)
        # Since r can be zero, we have to take care to treat that case
        # separately so as not to raise a numpy warning
        z = np.ones(r.shape)  # still true in case of obscuration
        rt = np.pi * r[r > 0]

        # Unobscured Airy equation
#        z[r > 0] = (2.0 * cls._j1(rt) / rt)**2

        # Obscured Airy equation ala wikipedia
#        z[r > 0] = 1.0/(1.0-obsrat**2)**2 *
#                   (2.0 * cls._j1(rt) / rt - 2.0 * obsrat *
#                   cls._j1(obsrat*rt) / rt )**2

        # Eric Pantin's equation
        z[r > 0] = (1.0 / (1.0 - obsrat**2)**2
                    * 1.0 / np.pi / r[r > 0]**2
                    * (cls._j1(rt) - obsrat**2 * cls._j1(obsrat * rt))**2)

        z *= amplitude

        return z



# %%

# --- helper function to find matching slice in SKIRT model cubes.
def find_wlen_match(theofits, wlen):
    """
    Find the matching wavelength frame in a model cube for a given wavelength
    """

    # first find the corresponding SED file that should come with the cube:
    sedfile = theofits.replace("_dust.fits", "_sed.dat")
    # could also be that the total emission file was provided
    sedfile = sedfile.replace("_total.fits", "_sed.dat")

    #print(sedfile)

    sed = ascii.read(sedfile, format='no_header', delimiter=' ',
                         guess=False, comment='#')
    # print(sed.colnames)

    sedwlen = np.array(sed['col1'], dtype=float)

    ind = np.argmin(np.abs(wlen-sedwlen))
    # print(ind, sedwlen[ind])

    return(ind, sedwlen[ind])


# %%
def simulate_image(psffits=None, theofits=None, obsfits=None, distance=None,
                   extfits=None, wlen=None, pfov=None, filt=None, psfext=0,
                   obsext=0, bgstd=0, bgmed=0, silent=False,
                   posang=0, foi_as=4, outname=None, manscale=None,
                   fnucdiam_as=0.45, suffix=None, fitsize=None, outfolder='.',
                   writesimfits=False, writetheofits=False, writeallfits=False,
                   writefitplot=False, debug=False, returncmod=True,
                   fitpsf=False, saveplot=False, meastime=False):

    """
    Simulate a (VISIR) imaging observaion given a real image, a PSF reference
    image and a model image for a provided object distance and position angle

    Current constraints:
        - The routine assumes that the images are squares (x = y)
    """

    if meastime:
        tstart = time.time()

    psfim = fits.getdata(psffits, ext=psfext)
    psfhead = fits.getheader(psffits)

    # print(obsfits, obsext)
    # ==== 1. Load Observational and PSF data ====
    if obsfits is not None:
        obsim = fits.getdata(obsfits, ext=obsext)
        obshead = fits.getheader(obsfits)

        if wlen is None:
            wlen = float(obshead['WAVELEN'])

        if pfov is None:
            pfov = obshead['PFOV']

        if filt is None:
            filt = obshead['Filter']

    else:
        if wlen is None:
            wlen = float(psfhead['WAVELEN'])

        if pfov is None:
            pfov = psfhead['PFOV']

        if filt is None:
            filt = psfhead['Filter']

    wlenstr = "{:.1f}".format(wlen)
    diststr = "{:.1f}".format(distance)
    pastr = "{:.0f}".format(posang)

    # --- optinal cropping of the PSF image
    if fitsize is not None:
        psfim = _crop_image(psfim, box=fitsize,
                            cenpos=_get_pointsource(psfim)[0][2:4])

    psfsize = np.array(np.shape(psfim))
    psfsize_as = psfsize[0] * pfov

    if not silent:
        print("Obs. wavelength [um]: "+ wlenstr)

    # ==== 2. Prepare theoretical image for convolution ====
    theohdu = fits.open(theofits)
    theohead = theohdu[0].header

    # --- check if provided file is the full cube
    if 'NAXIS3' in theohead:
        # find the right frame to be extracted
        mind, mwlen = find_wlen_match(theofits, wlen)
        theoim = theohdu[0].data[mind]

        if not silent:
            print(" - Model wavelength [um] | frame no: "+str(mwlen)+ " | "
                  + str(mind))

    else:
        theoim = theohdu[0].data

    theohdu.close()

    if meastime:
        print(" - All files read. Elapsed time: ", time.time()-tstart)

    if debug is True:
        print("THEOIM: ")
        plt.imshow(theoim, origin='bottom', norm=LogNorm(),
                   interpolation='nearest')
        plt.show()

    # --- if a manual scaling (fudge) factor was provided, apply to the model
    if manscale:
        theoim = theoim * manscale

    # --- rotate the model image (this changes its extent and thus has to be
    #     done first)
    # unrot = np.copy(theoim)
    if np.abs(posang-0) > 0.01:
        theoim = ndimage.interpolation.rotate(theoim, 180 - posang, order=0)


    if meastime:
        print(" - Model rotated. Elapsed time: ", time.time()-tstart)

    if debug is True:
        print("THEOIM_ROT: ")
        plt.imshow(theoim, origin='bottom', norm=LogNorm(),
                       interpolation='nearest')
        plt.show()

    # --- size units of the theoretical image
    theopixsize_pc = theohead['CDELT1']
    theosize = np.array(np.shape(theoim))
    theosize_pc = theopixsize_pc * theosize[0]  # pc
    theosize_as = 2 * np.arctan(theosize_pc/distance/2.0e6) * 180/np.pi * 3600
    theopixsize_as = (2 * np.arctan(theopixsize_pc / distance / 2.e6)
                      * 180/np.pi * 3600)

#    plt.imshow(unrot, origin='bottom', norm=LogNorm())
#    plt.imshow(theoim, origin='bottom', norm=LogNorm())
#    plt.imshow(theoim, origin='bottom')

#    theopfov = angsize/theosize[0]

    # normalize image to flux = 1
    # theoim = theoim/np.sum(theoim)
    # theoim = theoim/np.max(theoim)

    # --- convert to the right flux units
    # surface brightness unit in cube is W/m^2/arcsec2
    # -> convert to flux density in mJy
    # print(np.sum(theoim))
    freq = 2.99793e8 / (1e-6 * wlen)
    # print(freq)

    theoim = theoim * 1.0e29 / freq * theopixsize_as**2

    theototflux = np.sum(theoim)

    # --- resample to the same pixel size as the observation
    # print(   "- Resampling the model image to instrument pixel size...")
    # --- the ndimage.zoom sometimes creates weird artifacts and thus might not
    #     be a good choice. Instead, use the self-made routine
    # theoim_resres = ndimage.zoom(theoim_res, theopixsize_as/pfov, order=0)
    # --- do the rasmpling before scaling it to the size of the observation to
    #     save computing time
    theoim_res, sizerat = _increase_pixelsize(theoim,
                                             oldpfov=theopixsize_as,
                                             newpfov=pfov,
                                             meastime=False)

    if meastime:
        print(" - Pixelsize increased. Elapsed time: ", time.time()-tstart)

    if debug is True:
        print("sizerat: ", sizerat)
        print("THEOIM_RESRES: ")
        plt.imshow(theoim_res, origin='bottom', norm=LogNorm(),
                   interpolation='nearest')
        plt.show()


    # --- if the PSF frame size is larger than the model extend the model frame
    framerat = psfsize_as / theosize_as

    if debug is True:
        print("psfsize_as: ",psfsize_as)
        print("theosize_as: ",theosize_as)
        print("theopixsize_as: ",theopixsize_as)
        print("theosize_pc: ",theosize_pc)
        print("theosize: ",theosize)
        print("framerat: ", framerat)
    #print("newsize: ", newsize)

    if framerat > 1.0:
        theoim_resres = _extend_image(theoim_res, fac=framerat)

        if meastime:
            print(" - Frame extended. Elapsed time: ", time.time()-tstart)

        if debug is True:
            print("thesize_ext: ", np.shape(theoim_resres))
            print("THEOIM_RESRES: ")
            plt.imshow(theoim_resres, origin='bottom', norm=LogNorm(),
                       interpolation='nearest')
            plt.show()
        # plt.imshow(theoim_res, origin='bottom')

    else:
        theoim_resres = np.copy(theoim_res)


    theosize_new = np.array(np.shape(theoim_resres))

    if debug is True:
        print("theosize_new: ", theosize_new)
        print("new theosize_as: ", theosize_new * pfov)

    # print(np.sum(theoim_resres))

    # --- after resampling we need to re-establish the right flux levels
    theoim_resres = theoim_resres / np.sum(theoim_resres) * theototflux

    # plt.imshow(theoim_resres, origin='bottom', norm=LogNorm())
    # plt.imshow(theoim_resres, origin='bottom')

    # ==== 4. Optionally, apply a foreground extinction map if provided ====
    if extfits:

        exthdu = fits.open(extfits)
        exthead = exthdu[0].header
        extpfov = exthead["PFOV"]

        # check if provided file is the full cube
        if 'NAXIS3' in exthead:

            # find the right frame to be extracted
            if not mind:
                mind, mwlen = find_wlen_match(theofits, wlen)

            extim = exthdu[0].data[mind]

        else:
            extim = exthdu[0].data

        # plt.imshow(extim,origin='bottom')
        # plt.show()

        exthdu.close()

        # --- Do we have to resample the extinction map
        if np.abs(extpfov-pfov) > 0.001:
            # print("Resampling extinction map... ")
            extim = ndimage.zoom(extim, 1.0*pfov/extpfov, order=0)

        # --- match the size of the extinction map to the one of the model
        # image
        extsize = np.array(np.shape(extim))

        if theosize_new[0] > extsize[0]:
            print(" - ERROR: provided extinction map is too small to cover \
                   the imaged region. Abort...")
            return()

        if extsize[0] > theosize_new[0]:
            # print("Cuttting exctinction map...")
            extim = _crop_image(extim, box=theosize_new)

        # --- finally apply the extinction map
        theoim_resres = theoim_resres * extim
        # print('Maximum extinction: ',np.min(extim))

    # ==== 5. Obtain the Airy function from the calibrator ====
    if debug is True:
        print("PSF_IM: ")
        plt.imshow(psfim, origin='bottom', norm=LogNorm(),
                       interpolation='nearest')
        plt.show()

    # --- set up the fitting of the STD star image
    # --- assume that the maximum of the (cropped) image is the STD star
    if fitpsf is True:
        psfmax = np.max(psfim)
        psfmaxpos = np.unravel_index(np.argmax(psfim), psfsize)
        # print(im_bg, im_max, im_size, im_maxpos)

        # --- Use an Airy plus a Moffat + constant as PSF model
        c_init = models.Const2D(amplitude=np.median(psfim))

        oa_init = obsc_AiryDisk2D(amplitude=psfmax, x_0=psfmaxpos[1],
                                  y_0=psfmaxpos[0], radius=5, obsrat=0.15)

        m_init = models.Moffat2D(amplitude=psfmax, x_0=psfmaxpos[1],
                                 y_0=psfmaxpos[0], gamma=5, alpha=1)

        a_plus_m = oa_init + m_init + c_init

        # print(psfmax, psfmaxpos)

        # --- Selection of fitting method
        fit_meth = fitting.LevMarLSQFitter()

        # --- Do the Fitting:
        y, x = np.mgrid[:psfsize[0], :psfsize[1]]
        am_fit = fit_meth(a_plus_m, x, y, psfim)
        # print(am_fit.radius_0.value, am_fit.obsrat_0.value)

        # then refine the fit by subtracting another Moffat
        # a_plus_m_minus_m = am_fit - m_init
        # amm_fit = fit_meth(a_plus_m_minus_m, x, y, psfim)
        # z_amm = amm_fit(x, y)
#        res = psfim - z_amm
        #print(amm_fit.radius_0.value, amm_fit.obsrat_0.value)

        if meastime:
            print(" - PSf fit. Elapsed time: ", time.time()-tstart)

        if debug is True:
            print("PSF_FIT: ")
            plt.imshow(am_fit(x, y), origin='bottom', norm=LogNorm(),
                           interpolation='nearest')
            plt.show()

        # --- Genarate the PSF image with the fitted model
        # --- check whether the model image is on sky larger than the PSF image
        #     and if this is the case increase the grid and adjust the positions
        #     of the fits
        size_diff = theosize_new[0] - psfsize[0]
        if size_diff > 0:
            y, x = np.mgrid[:theosize_new[0], :theosize_new[1]]
            am_fit.x_0_0 = am_fit.x_0_0 + 0.5*size_diff
            am_fit.y_0_0 = am_fit.y_0_0 + 0.5*size_diff
            am_fit.x_0_1 = am_fit.x_0_1 + 0.5*size_diff
            am_fit.y_0_1 = am_fit.y_0_1 + 0.5*size_diff

        # --- create the final PSF image by subtracting the constant terms
        # and normalise by its area
        psf = am_fit(x, y) - am_fit.amplitude_2.value
        psf = psf / np.sum(psf)

        if debug is True:
            print("PSF_FINAL: ")
            plt.imshow(psf, origin='bottom', norm=LogNorm(),
                       interpolation='nearest')
            plt.show()

        # plt.imshow(psf, origin='bottom', norm=LogNorm())
        # plt.show()
        if writeallfits is True:

            fout = psffits.replace('.fits', '_fit.fits')
            if 'OBS NAME' in psfhead:
                psfhead.remove('OBS NAME')
            fits.writeto(fout, psf, psfhead, overwrite=True)

        # --- optinally, write the a plot documenting the quality of the PSF fit
        if writefitplot:

            fitplotfname = psffits.split('/')[-1]
            fitplotfname = fitplotfname.replace('.fits', '_fitplot.pdf')
            fitplotfname = outfolder + '/' + fitplotfname

            maxrad = int(0.5 * np.sqrt(0.5) * np.min(psfsize))

            _make_fit_plots(psfim, am_fit(x, y), fitplotfname, am_fit.x_0_0,
                           am_fit.y_0_0, maxrad, inv=True, cmap='gist_heat',
                           labelcolor='black')

    # --- alternatively the PSF can also be direclty provided so that no fit is
    #     necessary
    else:
        # normalise the provided psf
        psf = psfim - np.nanmin(psfim)
        psf = psf / np.nansum(psf)
        psf[np.argwhere(np.isnan(psf))] = 0


    # ==== 6. Convolution of the model image with the PSF ====
    simim = scipy.signal.convolve2d(theoim_resres, psf, mode='same')
    # print(np.sum(simim))

    if meastime:
        print(" - Model convolved. Elapsed time: ", time.time()-tstart)

    if debug is True:
        print("SIMIM:")
        plt.imshow(simim, origin='bottom', norm=LogNorm(),
                   interpolation='nearest')
        plt.show()


    # ==== 7. Flux measurements and application of noise ====
    # --- Flux measurement on the model image before applying noise
    ftotmod = int(np.nansum(simim))

    apos = 0.5 * np.array(theosize_new)

    nucrad_px = 0.5 * fnucdiam_as / pfov
    from aper import aper
    (mag, magerr, flux, fluxerr, sky, skyerr, badflag,
     outstr) = aper(simim, apos[1], apos[0], apr=nucrad_px,
                    exact=True, setskyval=0)

    fnucmod = int(flux)

    if not silent:
        print(' - Model total | nuclear flux [mJy]:    '+str(ftotmod)+' | '
              +str(fnucmod))


    plotsize_px = int(1.0 * foi_as / pfov)

    # --- Measure the background noise from the real observation
    if obsfits is not None:
        bg = np.copy(obsim)
    #    bgsize = np.shape(bg)
        params, _, _ = _get_pointsource(obsim)
        xpos = params[2]
        ypos = params[3]



        bg[int(ypos - 0.5 * plotsize_px) : int(ypos + 0.5 * plotsize_px),
           int(xpos - 0.5 * plotsize_px) : int(xpos + 0.5 * plotsize_px)] = 0

        if bgstd == 0:
            bgstd = np.std(bg[bg != 0])

        if bgmed == 0:
            bgmed = np.median(bg[bg != 0])
        # plt.imshow(bg, origin='bottom', norm=LogNorm(), cmap='gist_heat')

        # --- crop the observational image to the requested plot size
        cim = _crop_image(obsim, box=plotsize_px,
                          cenpos=_get_pointsource(obsim)[0][2:4])
        newsize = np.shape(cim)

        # --- make flux measurements on the nucleus and total
        ftotobs = int(np.sum(cim - bgmed))
        apos = 0.5 * np.array(newsize)

        (mag, magerr, flux, fluxerr, sky, skyerr, badflag,
         outstr) = aper(cim, apos[1], apos[0], apr=nucrad_px,
                        exact=True, setskyval=0)

        fnucobs = int(flux)

        if not silent:
            print(' - Observed total | nuclear flux [mJy]: '+str(ftotobs)+' | '
                  +str(fnucobs))

    else:
        cim = None

    if meastime:
        print(" - Fluxes measured. Elapsed time: ", time.time()-tstart)

    # pdb.set_trace()
    # --- crop the simulated image to the requested plot size
    params, _, _ = _get_pointsource(simim)
    csimim = _crop_image(simim, box=plotsize_px,
                         cenpos=_get_pointsource(simim)[0][2:4])
    if debug is True:
        plt.imshow(simim, origin='bottom', cmap='gist_heat', norm=LogNorm())
        plt.title('simim')
        plt.show()
        print(plotsize_px,np.shape(csimim))
        plt.imshow(csimim, origin='bottom', cmap='gist_heat', norm=LogNorm())
        plt.title('csimim')
        plt.show()

    if meastime:
        print(" - Obs cropped. Elapsed time: ", time.time()-tstart)

    # --- generate an artifical noise frame with same properties as in real
    #     image
    if bgstd > 0:
        artbg = np.random.normal(scale=bgstd, size=(plotsize_px, plotsize_px)) + bgmed
        # artbg = 0

        # --- apply the artificial background
        # csimim = csimim/np.max(csimim)*(np.max(cim)-bgmed)+bgmed+artbg
        csimim = csimim + artbg

    # --- crop the PSF image to the requested plot size
    cpsf = _crop_image(psf, box=plotsize_px,
                       cenpos=_get_pointsource(psf)[0][2:4])

    # print(bgmed, bgstd, np.std(artbg), np.median(cim), np.median(csimim),
    # np.std(cim), np.std(csimim), np.max(cim), np.min(cim), np.max(csimim),
    # np.min(csimim))

    if meastime:
        print(" - PSF cropped. Elapsed time: ", time.time()-tstart)

    theopfov = theopixsize_as
    theobox = int(np.round(plotsize_px * pfov/theopfov))

    if returncmod:
        # --- crop the model imae to the requested plot size
        if framerat > 1.0:
            theoim_res_0 = _extend_image(theoim, fac=framerat)
        else:
            theoim_res_0 = theoim
        cmod = _crop_image(theoim_res_0, box=theobox, exact=False)
        # plt.imshow(cmod, origin='bottom', norm=LogNorm(), cmap='gist_heat')

        if meastime:
            print(" - Theo cropped. Elapsed time: ", time.time()-tstart)

    else:
        cmod = None

    # --- write out the simulated image as a fits file
    if not outname:

        theofitsfile = theofits.split("/")[-1]

        out_str = theofitsfile.replace("_total.fits", "")

        out_str = (out_str + "_pa" + pastr + "_dist" + diststr + "_wlen"
                   + wlenstr)

        if suffix:
            out_str = out_str + "_" + suffix

    else:
        out_str = outname

    out_str = outfolder + '/' + out_str

    if writeallfits or writetheofits:

        fout = out_str + '_mod.fits'

        theohead["Filter"] = filt
        theohead["WAVELEN"] = wlen
        theohead["PFOV"] = theopixsize_as

        fits.writeto(fout, cmod, theohead, overwrite=True)

        if saveplot:
            _simple_image_plot(cmod, fout.replace(".fits", ".png"), log=True)

    if writeallfits or writesimfits:

        fout = out_str + '_sim.fits'
        theohead["PFOV"] = pfov
        theohead["BUNIT"] = 'mJy'

        ts = np.shape(simim)
        theohead["CRPIX1"] =  0.5*ts[1]
        theohead["CDELT1"] = pfov
        theohead["CTYPE1"] = 'arcsec'
        theohead["CRPIX2"] =  0.5*ts[0]
        theohead["CDELT2"] = pfov
        theohead["CTYPE2"] = 'arcsec'

        fits.writeto(fout, csimim, theohead, overwrite=True)

        if saveplot:
            _simple_image_plot(csimim, fout.replace(".fits", ".png"), log=True)

    if meastime:
        print(" - All finished: ", time.time()-tstart)

    return(cpsf, cmod, cim, csimim, wlen, pfov, theopfov)


# %%
def simulate_images(psffits, theofits, distance, obsfits=None, extfits=None,
                    posang=0, foi_as=4, fnucdiam_as=0.45, outname=None,
                    scale='log', inv=True, cmap='gist_heat', manscale=None,
                    suffix=None, fitsize=None, outfolder='.',
                    labelcolor='black', writeoutfits=False,
                    writefitplot=False, debug=False, fitpsf=False,
                    theomax=99.99, theomin=0.01, wavelens=None, pfovs=None,
                    filts=None, immin=None, immax=None, theoscale='log'):

    # --- TO DO:
    #  - Writefitplot currently collides with the main plot and makes it empty
    #    for unknown reasons


    # --- check whether a list (or just a single file) was provided
    if not isinstance(psffits,list):
        psffits = [psffits]


    nf = len(psffits)

    if obsfits is not None:
        if not isinstance(obsfits,list):
            obsfits = [obsfits]

    else:
        obsfits = [None] * nf

    if wavelens is None:
        wavelens = [None] * nf

    if pfovs is None:
        pfovs = [None] * nf

    if filts is None:
        filts = [None] * nf

    # --- prepare the plot
    if debug is False:

        plt.clf()
        mpl.rcdefaults()
        # fig = plt.figure(1, (10, 24))
        fig = plt.figure(figsize=(16, 1 + 4*nf))
        fig.subplots_adjust(bottom=0.2)
        # ax = fig.add_subplot(111)

        if inv:
            cmap = cmap+'_r'

        mpl.rc('axes', edgecolor=labelcolor)
        mpl.rc('xtick', color=labelcolor)
        mpl.rc('ytick', color=labelcolor)

        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(nf, 4),  # creates 2x2 grid of axes
                         axes_pad=0.1, aspect=True  # pad between axes in inch.
                         )

    wliststr = ''

    # --- simulate the individual images
    for i in range(nf):

        (psf, mod, im, simim, wlen, pfov,
         theopfov) = simulate_image(psffits=psffits[i], theofits=theofits,
                                    obsfits=obsfits[i],
                                    distance=distance, extfits=extfits,
                                    posang=posang, foi_as=foi_as,
                                    fnucdiam_as=fnucdiam_as, outname=outname,
                                    manscale=manscale, suffix=suffix,
                                    fitsize=fitsize, outfolder=outfolder,
                                    writeallfits=writeoutfits, fitpsf=fitpsf,
                                    writefitplot=writefitplot, debug=debug,
                                    wlen=wavelens[i], pfov=pfovs[i],
                                    filt=filts[i])
        print("")


        # -- set the ranges
        if immin is not None:
            minval = np.nanpercentile(im, immin)
            im[im < minval] = minval
            simim[simim < minval] = minval

        if immax is not None:
            maxval = np.nanpercentile(im, immax)
            im[im > maxval] = maxval
            simim[simim > maxval] = maxval

        # --- cut the low background noise
#        im[im < np.median(im)] = np.median(im)
#        simim[simim < np.median(simim)] = np.median(simim)

        # --- scale the model image differently
        if theomin is not None:
            theomin_val = np.nanpercentile(mod[mod > 0], theomin)
            mod[mod < theomin_val] = theomin_val

        if theomax is not None:
            theomax_val = np.nanpercentile(mod[mod > 0], theomax)
            mod[mod > theomax_val] = theomax_val


        if theoscale == 'log':
            mod = np.log10(1000 * (mod - np.nanmin(mod)) / np.nanmax(mod) + 1)
        elif theoscale ==  'asinh':
            mod = np.arcsinh(mod)

        #print(theomin_val, theomin)

        if scale == 'log':

            norm = np.max(im)
            psf = np.log10(1000 * (psf - np.min(psf)) / np.max(psf) + 1)
            im = np.log10(1000 * (im - np.min(im)) / norm + 1)
            simim = np.log10(1000 * (simim - np.min(simim)) / norm + 1)

        # print(np.min(im))

        # print(np.nanmin(mod))
        if scale == 'sqrt':

            psf = np.sqrt(psf)
            im = np.sqrt(im)
            simim = np.sqrt(simim)
            mod = np.sqrt(mod)

        if scale == 'asinh':

            psf = np.arcsinh(psf)
            im = np.arcsinh(im)
            simim = np.arcsinh(simim)


        # dirty trick to ensure that the simulated image has the same scaling
        # as the real image
        simim[0, 0] = np.max(im)

        # --- do the actual plotting
        if debug is False:

            handle = grid[0 + i*4].imshow(psf, cmap=cmap, origin='lower',
                                          interpolation='nearest')

            grid[0 + i*4].set_title('PSF', x=0.5, y=0.85, color=labelcolor)

            wlenstr = "{:.1f}".format(wlen)
            grid[0 + i*4].text(0.05, 0.95, wlenstr + '$\mu\mathrm{m}$',
                                fontsize=20, transform=grid[0 + i*4].transAxes,
                                verticalalignment='top',
                                horizontalalignment='left')

            ny, nx = im.shape
            handle.set_extent(np.array([-nx/2, nx/2 - 1, -nx/2,
                                        nx / 2 - 1]) * pfov)

            handle = grid[1 + i*4].imshow(im, cmap=cmap, origin='lower',
                                          interpolation='nearest')

            grid[1 + i*4].set_title('real', x=0.5, y=0.85, color=labelcolor)
            handle.set_extent(np.array([-nx/2, nx/2 - 1, -nx/2,
                                        nx / 2 - 1]) * pfov)

            handle = grid[2 + i*4].imshow(simim, cmap=cmap, origin='lower',
                                          interpolation='nearest')
            grid[2 + i*4].set_title('simulated', x=0.5, y=0.85,
                                    color=labelcolor)
            handle.set_extent(np.array([-nx/2, nx/2 - 1, -nx/2,
                                        nx / 2 - 1]) * pfov)

            handle = grid[3 + i*4].imshow(mod, cmap=cmap, origin='lower',
                                          interpolation='nearest')
            mny, mnx = mod.shape
            grid[3 + i*4].set_title('model', x=0.5, y=0.85, color=labelcolor)
            handle.set_extent(np.array([-mnx/2, mnx/2 - 1, -mnx/2,
                                        mnx / 2 - 1]) * theopfov)

            wliststr = wliststr + wlenstr + "+"

            # --- formatting of axis
            # get the extent of the largest box containing all the axes/subplots
            extents = np.array([bla.get_position().extents for bla in grid])
            bigextents = np.empty(4)
            bigextents[:2] = extents[:, :2].min(axis=0)
            bigextents[2:] = extents[:, 2:].max(axis=0)

            # --- text to mimic the x and y label. The text is positioned in
            #     the middle
            xlabelpad = 0.03  # distance between the external axis and the text
            ylabelpad = 0.03
            fig.text((bigextents[2] + bigextents[0])/2,
                     bigextents[1] - xlabelpad, 'RA offset ["]',
                     horizontalalignment='center', verticalalignment='bottom')
            fig.text(bigextents[0] - ylabelpad,
                     (bigextents[3] + bigextents[1])/2, 'DEC offset ["]',
                     rotation='vertical', horizontalalignment='left',
                     verticalalignment='center')


#plt.show()
    if debug is False:

        if not outname:

            theofitsfile = theofits.split("/")[-1]

            out_str = theofitsfile.replace(".fits", "")

            diststr = "{:.1f}".format(distance)
            pastr = "{:.0f}".format(posang)
            wliststr = wliststr[0:-1]

            out_str = (out_str + "_pa" + pastr + "_dist" + diststr + "_wlen"
                       + wliststr)

            if suffix:
                out_str = out_str + "_" + suffix

        else:
            out_str = outname

        out_str = outfolder + '/' + out_str

        imout = out_str + ".pdf"
        plt.savefig(imout, bbox_inches='tight', pad_inches=0.1)

