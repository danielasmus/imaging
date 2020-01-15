#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-15: created by Daniel Asmus


NOTES:
    - Collection of python routines for handling astrophysical imaging data

TO-DO:
    -
"""


import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

from .create_azimuthal_average import create_azimuthal_average
from .create_alpha_colmap import create_alpha_colmap
from .crop_image import crop_image
from .crosshair_marker import crosshair_marker
from .extend_image import extend_image
from .get_background import get_background
from .get_pointsource import get_pointsource
from .get_shift import get_shift
from .homogenize_image import homogenize_image
from .increase_pixelsize import increase_pixelsize
from .make_ext_measurements import make_ext_measurements
from .make_fit_plots import make_fit_plots
from .make_gallery import make_gallery
from .radial_distr import radial_distr
from .simple_image_plot import simple_image_plot
from .simulate_images import simulate_image, simulate_images
from .subtract_source import subtract_source

