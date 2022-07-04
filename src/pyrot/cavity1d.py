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

    def scattering_matrix(self, omega, zero_offset=0.0):
        return parratt_maxwell1D_matrix(self.n, self.t, omega, phase_zero_offset=-omega*zero_offset)

    def transmission_coefficient(self, omega, input_from_right=False, zero_offset=0.0):
        if input_from_right:
            return self.scattering_matrix(omega, zero_offset=zero_offset)[1,1]
        return self.scattering_matrix(omega, zero_offset=zero_offset)[0,0]

    def transmission_intensity(self, omega, input_from_right=False, zero_offset=0.0):
        return np.abs(self.transmission_coefficient(omega, input_from_right=input_from_right, zero_offset=zero_offset))**2

    def reflection_coefficient(self, omega, input_from_right=False, zero_offset=0.0):
        if input_from_right:
            return self.scattering_matrix(omega, zero_offset=zero_offset)[0,1]
        return self.scattering_matrix(omega, zero_offset=zero_offset)[1,0]

    def reflection_intensity(self, omega, input_from_right=False, zero_offset=0.0):
        return np.abs(self.reflection_coefficient(omega, input_from_right=input_from_right, zero_offset=zero_offset))**2

    def green_function(self, z1, z2, omega):
        Theta = np.pi/2. # 1D cavity corresponds to normal incidence of layer stack
        ### loop in case of list/array omega: ###
        if hasattr(omega, "__len__"):
            result = np.empty((len(z1), len(z2), len(omega)), dtype=np.complex128)
            for i, om_ in enumerate(omega):
                result[:,:,i] = GF(z1, z2, self, Theta, om_) # TODO: numpify this loop by making GF recursion algorithm accept omega arrays
            return result
        ### normal call for single value omega: ###
        return GF(z1, z2, self, Theta, omega)


    def draw_cav(self, depth, loc=4):
        N_depth = self.n_depth(depth)

        plt.figure()
        plt.xlabel('Depth')
        plt.ylabel('Refractive index')
        plt.title('Example cavity sketch')
        plt.plot(depth, np.real(N_depth), '-', label='Re[N]')
        plt.plot(depth, np.imag(N_depth), '-', label='Im[N]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend(loc=loc,fontsize=8)
        plt.show()

################################################################################
### algorithmic functions: Parratt's formalism ###

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

##########################################################################################################
### algorithmic functions: Green's function
### (from Tomas1995: https://doi.org/10.1103/PhysRevA.51.2545)

def j_from_z(z, cav): # z in [L]
    """
    Convert depth from cavity surface into layer index + depth from layer surface.

    Returns the layer index j and depth from the layer boundary z-z_j
    given the total depth z and a cavity cav.
        - j=0 corresponds to the first layer (vacuum in pynuss) where z<0.
          The distance to the upper layer boundary is not defined in this case
          and given as -z (TODO: implement output interface for fields and check if field formula applies in outside region)
        - j=1 is the first layer, with layer boundary position z_1 = 0
        - j=2 is the second layer, with layer boundary position z_2 = t_1
          (t_1: thickness of the first layer)
        - j>2 is treated analogously.
    """
    Thicknesses = cav.t[1:]
    if z<0.:
        return 0, z
    if z==0.:
        return 1, z
    for j, t in enumerate(Thicknesses[0:-1]):
        if ( sum(Thicknesses[0:j]) < z ) and ( sum(Thicknesses[0:j+1]) >= z):
            return j+1, z-np.sum(Thicknesses[0:j])
    return j+2, z-np.sum(Thicknesses[0:j+1]) # returns index and sum of layer thicknesses above in [L]

def Εs_0(z, cav, Theta, omega):
    Field = np.zeros_like(z, dtype=np.complex128)
    N, T = cav.n, cav.t
    for i,zi in enumerate(z):
        n = len(N)-1
        j, z_offset = j_from_z(zi, cav)
        betaj = beta_j(j, N, T, Theta, omega)
        dj = cav.t[j] # [L]
        if j==0 or j==(len(N)-1):
            dj = 0.
        rs_j0 = r_i_j(j, 0, N, T, Theta, omega, pol='s') # = rs_j-
        rs_jn = r_i_j(j, n, N, T, Theta, omega, pol='s') # = rs_j+
        ts_0j = t_i_j(0, j, N, T, Theta, omega, pol='s')
        Dsj = 1. - rs_j0 * rs_jn * np.exp(2j*betaj*dj)
        zm = z_offset
        zp = dj - z_offset
        Field[i] = ts_0j*np.exp(1j*betaj*dj)/Dsj * ( np.exp(-1j*betaj*zp) +  rs_jn*np.exp(+1j*betaj*zp) )
    return z, Field

def Εs_n(z, cav, Theta, omega):
    Field = np.zeros_like(z, dtype=np.complex128)
    N, T = cav.n, cav.t
    for i,zi in enumerate(z):
        n = len(N)-1
        j, z_offset = j_from_z(zi, cav)
        betaj = beta_j(j, N, T, Theta, omega)
        dj = cav.t[j] # [L]
        if j==0 or j==(len(N)-1):
            dj = 0.
        rs_j0 = r_i_j(j, 0, N, T, Theta, omega, pol='s') # = rs_j-
        rs_jn = r_i_j(j, n, N, T, Theta, omega, pol='s') # = rs_j+
        ts_nj = t_i_j(n, j, N, T, Theta, omega, pol='s')
        Dsj = 1. - rs_j0 * rs_jn * np.exp(2j*betaj*dj)
        zm = z_offset
        zp = dj - z_offset
        Field[i] = ts_nj*np.exp(1j*betaj*dj)/Dsj * ( np.exp(-1j*betaj*zm) +  rs_j0*np.exp(+1j*betaj*zm) )
    return z, Field

def beta_j(j, N, T, Theta, omega):
    k = omega # [1/L]
    k_parallel = k*np.cos(Theta) # [1/L]
    betaj = np.sqrt(N[j]**2*k**2-k_parallel**2)
    return betaj # [1/L]

def D_j_i_k(j, i, k, N, T, Theta, omega, pol='s'):
    betaj = beta_j(j, N, T, Theta, omega)
    dj = T[j] # [m]
    rj_i = r_i_j(j, i, N, T, Theta, omega, pol=pol)
    rj_k = r_i_j(j, k, N, T, Theta, omega, pol=pol)
    return 1. - rj_i*rj_k*np.exp(2.j*betaj*dj)

def gamma_ij(i, j, N, T, Theta, omega, pol='s'):
    ### single interface, abs(i-j)=1 ###
    if not (np.abs(i-j) == 1):
        raise ValueError('Not adjacent layers, gamma_ij not defined.')
    if pol=='s':
        return 1.+0.j
    ϵi = N[i]**2
    ϵj = N[j]**2
    return ϵi/ϵj

def r_ij(i, j, N, T, Theta, omega, pol='s'):
    if not (np.abs(i-j) == 1):
        raise ValueError('Not adjacent layers, r_ij not defined.')
    betai = beta_j(i, N, T, Theta, omega)
    betaj = beta_j(j, N, T, Theta, omega)
    gammaij = gamma_ij(i, j, N, T, Theta, omega, pol=pol)
    return (betai - gammaij*betaj)/(betai + gammaij*betaj)

def t_ij(i, j, N, T, Theta, omega, pol='s'):
    if not (np.abs(i-j) == 1):
        raise ValueError('Not adjacent layers, t_ij not defined.')
    gammaij = gamma_ij(i, j, N, T, Theta, omega, pol=pol)
    rij = r_ij(i, j, N, T, Theta, omega, pol=pol)
    return np.sqrt(gammaij)*(1. + rij)

def r_i_j_k(i, j, k, N, T, Theta, omega, pol='s'):
    ### recurrence relation ###
    betaj = beta_j(j, N, T, Theta, omega)
    dj = T[j] # [m]
    Dj = D_j_i_k(j, i, k, N, T, Theta, omega, pol=pol)
    ri_j = r_i_j(i, j, N, T, Theta, omega, pol=pol)
    rj_i = r_i_j(j, i, N, T, Theta, omega, pol=pol)
    rj_k = r_i_j(j, k, N, T, Theta, omega, pol=pol)
    ti_j = t_i_j(i, j, N, T, Theta, omega, pol=pol)
    tj_i = t_i_j(j, i, N, T, Theta, omega, pol=pol)
    return 1./Dj * ( ri_j + (ti_j*tj_i - ri_j*rj_i) * rj_k * np.exp(2j*betaj*dj) )

def t_i_j_k(i, j, k, N, T, Theta, omega, pol='s'):
    ### recurrence relation ###
    betaj = beta_j(j, N, T, Theta, omega)
    dj = T[j] # [L]
    Dj = D_j_i_k(j, i, k, N, T, Theta, omega, pol=pol)
    ti_j = t_i_j(i, j, N, T, Theta, omega, pol=pol)
    tj_k = t_i_j(j, k, N, T, Theta, omega, pol=pol)
    return 1./Dj * ti_j*tj_k * np.exp(1j*betaj*dj)

def r_i_j(i, j, N, T, Theta, omega, pol='s'):
    ### starts and ends the recurrence chain ###
    if np.abs(i-j) == 1:
        return r_ij(i, j, N, T, Theta, omega, pol=pol)
    if i==j:
        return 0.+0.j
    # choose middle index to start recurrence chain #
    if i>j:
        k=i-1
    else:
        k=i+1
    return r_i_j_k(i, k, j, N, T, Theta, omega, pol=pol)

def t_i_j(i, j, N, T, Theta, omega, pol='s'):
    ### starts and ends the recurrence chain ###
    if np.abs(i-j) == 1:
        return t_ij(i, j, N, T, Theta, omega, pol=pol)
    if i==j:
        return 1.+0.j
    # choose middle index to start recurrence chain #
    if i>j:
        k=i-1
    else:
        k=i+1
    return t_i_j_k(i, k, j, N, T, Theta, omega, pol=pol)

def GF(z, z0, cav, Theta, omega):
    """
        Only implements the s-polarization part,
        which is the relevant component for a 1D
        cavity at normal incidence.
    """
    N, T = cav.n, cav.t
    xis = -1
    n = len(N)-1
    betan = beta_j(n, N, T, Theta, omega)
    ts_0n = t_i_j(0, n, N, T, Theta, omega)
    # only single pol (s):
    zs,Es0_1 = Εs_0(z, cav, Theta, omega)
    zs,Esn_1 = Εs_n(z0, cav, Theta, omega)
    zs,Es0_2 = Εs_0(z0, cav, Theta, omega)
    zs,Esn_2 = Εs_n(z, cav, Theta, omega)
    Z0, Z = np.meshgrid(z0, z) # note reversed order for consistency with np.outer
    heavi_1 = np.heaviside(np.real(Z-Z0), 0.5)
    heavi_2 = np.heaviside(np.real(Z0-Z), 0.5)
    return 2j*np.pi/betan * xis/ts_0n * ( np.outer(Es0_1, Esn_1)*heavi_1 + np.outer(Esn_2, Es0_2)*heavi_2 )









