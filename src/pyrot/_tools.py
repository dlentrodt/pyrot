"""
Copyright (C) 2022 Dominik Lentrodt

This file is part of pyrot.

pyrot is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pyrot is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pyrot.  If not, see <http://www.gnu.org/licenses/>.
"""

import copy

import sys
import time

import math
import numpy as np
import numpy.linalg

import matplotlib
import matplotlib.pylab as plt

################################################################################
### helper functions ###

def find_nearest_idx(array, value):
    """
    Find the closest element in an array and return the corresponding index.
    """
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx

def find_nearest(array, value):
    """
    Find the closest element in an array and return the corresponding index and value.
    """
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]










