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

from ._tools import *

################################################################################

### main classes ###

class Cavity1d():
    def __init__(self, n, t):
        self.n = n
        self.t = t
        self.r0 = 0.0 # default zero position of spatial axis: at the left end of the cavity
        if (self.t[0] != -1) or (self.t[-1] != -1):
            raise ValueError('First and last element of T must be set to -1. \
                They indicate the spaces to the left and right of the cavity.')

    def n_depth(self, depth):
        N = self.n
        T = self.t
        N_depth = np.zeros_like(depth, dtype=np.complex128)
        layer_counters = np.zeros_like(depth)

        layer_counter = 0
        for i, d_ in enumerate(depth):
            if layer_counter == 0:
                N_depth[i] = 1.0 + 0.0j
                if d_>0:
                    layer_counter += 1
            elif layer_counter == len(N)-1:
                N_depth[i] = 1.0 + 0.0
            else:
                if d_>np.sum(T[1:layer_counter+1]):
                    layer_counter += 1
                N_depth[i] = N[layer_counter]
            layer_counters[i] = layer_counter
        return N_depth

    def scattering_matrix(self, k, zero_offset=0.0):
        return parratt_maxwell1D_matrix(self.n, self.t, k, phase_zero_offset=-k*zero_offset)

    def transmission_coefficient(self, k, input_from_right=False, zero_offset=0.0):
        if input_from_right:
            return self.scattering_matrix(k, zero_offset=zero_offset)[1,1]
        return self.scattering_matrix(k, zero_offset=zero_offset)[0,0]

    def transmission_intensity(self, k, input_from_right=False, zero_offset=0.0):
        return np.abs(self.transmission_coefficient(k, input_from_right=input_from_right, zero_offset=zero_offset))**2

    def reflection_coefficient(self, k, input_from_right=False, zero_offset=0.0):
        if input_from_right:
            return self.scattering_matrix(k, zero_offset=zero_offset)[1,0]
        return self.scattering_matrix(k, zero_offset=zero_offset)[0,1] # TODO: check order

    def reflection_intensity(self, k, input_from_right=False, zero_offset=0.0):
        return np.abs(self.reflection_coefficient(k, input_from_right=input_from_right, zero_offset=zero_offset))**2

    def draw_cav(self, depth):
        N_depth = self.n_depth(depth)

        plt.figure()
        plt.xlabel('Depth')
        plt.ylabel('Refractive index')
        plt.title('Example cavity sketch')
        plt.plot(depth, np.real(N_depth), '-', label='Re[N]')
        plt.plot(depth, np.imag(N_depth), '-', label='Im[N]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend(loc=4,fontsize=8)
        plt.show()

################################################################################
### algorithmic functions ###

def parratt_maxwell1D_matrix(N0, D0, kRange, phase_zero_offset=None):
    """ Calculates the scattering matrix of an empty 1D cavity (no atom, cavity alone)
        given by a refractive index profile of discrete layers.
        Input format:
        - N0: [N0, N1, N2, ..., NN, NN+1]; refractive indices of different layers,
                                           where N0,NN+1 is the space to the left/right,
                                           respectively and the rest are the layers.
        - D0: [-1, D1, D2, ..., DN, -1];   thicknesses of the different layers,
                                           left/right space do not have thicknesses, the rest are the layers
        - kRange: a 1D array/list of k-values or frequencies to compute the spectra on.
        - phase_zero_offset: The phase of hell... forget about this.
         NOTE: The elements of N0 can either be numbers or arrays.
               The latter corresponds to an energy dependence.
    """
    if not all(isinstance(x, float) for x in N0):
        return parratt_maxwell1D_matrix_eDep(N0, D0, kRange, phase_zero_offset=None)
    N = np.asarray(N0, dtype=np.complex128)
    D = np.asarray(D0, dtype=np.complex128)
    k = np.asarray(kRange, dtype=np.complex128)
    transfer_matrix_tot = np.asarray([[np.ones_like(k), np.zeros_like(k)],
                                      [np.zeros_like(k), np.ones_like(k)]])
    for p in np.arange(len(N)-1):
        n0 = N[p]
        n1 = N[p+1]
        kZ0 = k*np.sqrt(n0**2)
        kZ1 = k*np.sqrt(n1**2)
        r01 = (kZ0-kZ1)/(kZ0+kZ1)
        t01 = 2.0*np.sqrt(kZ0*kZ1)/(kZ0+kZ1)
        transfer_matrix_interface = np.asarray([[1.0/t01, r01/t01],
                                                [r01/t01, 1.0/t01]])
        if p<(len(N)-2):
            d1 =  D[p+1]
            phi1 = kZ1*d1
            transfer_matrix_layer = np.asarray([[np.exp(-1j*phi1), np.zeros_like(phi1)],
                                                [np.zeros_like(phi1), np.exp(1j*phi1)]])
            transfer_matrix_tot = np.einsum('ijk,jlk,lmk->imk', transfer_matrix_tot,
                                                                transfer_matrix_interface,
                                                                transfer_matrix_layer)
        else:
            transfer_matrix_tot = np.einsum('ijk,jlk->ilk', transfer_matrix_tot, transfer_matrix_interface)
    if not (phase_zero_offset is None):
        transfer_to_hell = np.asarray([[np.exp(-1j*phase_zero_offset), np.zeros_like(phi1)],
                                       [np.zeros_like(phi1), np.exp(1j*phase_zero_offset)]])
        transfer_matrix_tot = np.einsum('ijk,jlk->ilk', transfer_matrix_tot, transfer_to_hell)
    R1 = transfer_matrix_tot[1,0,:]/transfer_matrix_tot[0,0]
    R2 = -transfer_matrix_tot[0,1,:]/transfer_matrix_tot[0,0]
    T1 = 1.0/transfer_matrix_tot[0,0]
    T2 = -transfer_matrix_tot[1,0,:]*transfer_matrix_tot[0,1,:]/transfer_matrix_tot[0,0,:] \
             + transfer_matrix_tot[1,1,:]
    return np.asarray([[T1, R2], [R1, T2]])

def parratt_maxwell1D_matrix_eDep(N0, D0, kRange, phase_zero_offset=None):
    """
        Explicitly implements the energy dependent version of parratt_maxwell1D_matrix.
        # NOTE: Potential change for the future - change edep input from direct array to function.
    """
    D = np.asarray(D0, dtype=np.complex128)
    k = np.asarray(kRange, dtype=np.complex128)
    N = np.zeros((len(N0), len(k)), dtype=np.complex128) 
    for i, n_l in enumerate(N0):
        N[i, :] = n_l
    transfer_matrix_tot = np.asarray([[np.ones_like(k), np.zeros_like(k)],
                                      [np.zeros_like(k), np.ones_like(k)]])
    for p in np.arange(len(N)-1):
        n0 = N[p, :]
        n1 = N[p+1, :]
        kZ0 = k*np.sqrt(n0**2)
        kZ1 = k*np.sqrt(n1**2)
        r01 = (kZ0-kZ1)/(kZ0+kZ1)
        t01 = 2.0*np.sqrt(kZ0*kZ1)/(kZ0+kZ1)
        transfer_matrix_interface = np.asarray([[1.0/t01, r01/t01],
                                                [r01/t01, 1.0/t01]])
        if p<(len(D)-2):
            d1 =  D[p+1]
            phi1 = kZ1*d1
            transfer_matrix_layer = np.asarray([[np.exp(-1j*phi1), np.zeros_like(phi1)],
                                                [np.zeros_like(phi1), np.exp(1j*phi1)]])
            transfer_matrix_tot = np.einsum('ijk,jlk,lmk->imk', transfer_matrix_tot,
                                                                transfer_matrix_interface,
                                                                transfer_matrix_layer)
        else:
            transfer_matrix_tot = np.einsum('ijk,jlk->ilk', transfer_matrix_tot, transfer_matrix_interface)
    if not (phase_zero_offset is None):
        transfer_to_hell = np.asarray([[np.exp(-1j*phase_zero_offset), np.zeros_like(phi1)],
                                       [np.zeros_like(phi1), np.exp(1j*phase_zero_offset)]])
        transfer_matrix_tot = np.einsum('ijk,jlk->ilk', transfer_matrix_tot, transfer_to_hell)
    R1 = transfer_matrix_tot[1,0,:]/transfer_matrix_tot[0,0]
    R2 = -transfer_matrix_tot[0,1,:]/transfer_matrix_tot[0,0]
    T1 = 1.0/transfer_matrix_tot[0,0]
    T2 = -transfer_matrix_tot[1,0,:]*transfer_matrix_tot[0,1,:]/transfer_matrix_tot[0,0,:] \
             + transfer_matrix_tot[1,1,:]
    return np.asarray([[T1, R2], [R1, T2]])











