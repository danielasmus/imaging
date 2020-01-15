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
from matplotlib.path import Path

def crosshair_marker(inner_r=0.5, pa = 0):
   ''' The path of an emtpy cross, useful for indicating targets without crowding the field.

      inner_r = empty inner radius. Default =0.5
   '''

   verts = [(-1, 0),
            (-inner_r, 0),
            (0, inner_r),
            (0, 1),
            (inner_r, 0),
            (1, 0),
            (0, -inner_r),
            (0, -1),
            (-1, 0),
            (-1, 0),
           ]

   pa = np.radians(pa)
   rot_mat = np.matrix([[np.cos(pa),-np.sin(pa)],[np.sin(pa),np.cos(pa)]])

   for (v, vert) in enumerate(verts):
      verts[v] = (vert*rot_mat).A[0]

   codes = [Path.MOVETO,
            Path.LINETO,
            Path.MOVETO,
            Path.LINETO,
            Path.MOVETO,
            Path.LINETO,
            Path.MOVETO,
            Path.LINETO,
            Path.MOVETO,
            Path.CLOSEPOLY
            ]

   path = Path(verts, codes)

   return path
