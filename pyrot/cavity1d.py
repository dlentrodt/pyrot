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

class CavityAtoms1d(Cavity1d):

    def __init__(self, n, t, atoms_params):
        # atoms_params = [[atom1_pos, atom1_dpol, atom1_om, atom1_gamma],
        #                 [atom2_pos, atom2_dpol, atom2_om, atom2_gamma],
        #                 ... ]
        self.atoms_params = atoms_params
        if np.shape(self.atoms_params)[1] != 4:
            raise ValueError('atom_params does not have the right dimensions. Natoms x 4 required.')

        super().__init__(n, t)

    def linear_scattering_matrix(self, k, zero_offset=0.0):
        _, _, result = linear_dispersion_scattering_multi_atom(k, self.n, self.t, self.atoms_params, phase_zero_offset=-k*zero_offset)
        return result

    def linear_transmission_coefficient(self, k, input_from_right=False, zero_offset=0.0):
        if input_from_right:
            return self.linear_scattering_matrix(k, zero_offset=zero_offset)[1,1]
        return self.linear_scattering_matrix(k, zero_offset=zero_offset)[0,0]

    def linear_transmission_intensity(self, k, input_from_right=False, zero_offset=0.0):
        return np.abs(self.linear_transmission_coefficient(k, input_from_right=input_from_right, zero_offset=zero_offset))**2

    def linear_reflection_coefficient(self, k, input_from_right=False, zero_offset=0.0):
        if input_from_right:
            return self.linear_scattering_matrix(k, zero_offset=zero_offset)[1,0]
        return self.linear_scattering_matrix(k, zero_offset=zero_offset)[0,1] # TODO: check order

    def linear_reflection_intensity(self, k, input_from_right=False, zero_offset=0.0):
        return np.abs(self.linear_reflection_coefficient(k, input_from_right=input_from_right, zero_offset=zero_offset))**2

    def linear_layer_system_with_atom(self, k, zero_offset=0.0):
        N, T, _ = linear_dispersion_scattering_multi_atom(k, self.n, self.t, self.atoms_params, phase_zero_offset=-k*zero_offset)
        return N, T

    def draw_cav(self, depth):
        N_depth = self.n_depth(depth)

        plt.figure()
        plt.xlabel('Depth')
        plt.ylabel('Refractive index')
        plt.title('Example cavity sketch')
        plt.plot(depth, np.real(N_depth), '-', label='Re[N]')
        plt.plot(depth, np.imag(N_depth), '-', label='Im[N]')
        for atom_params in self.atoms_params:
            plt.axvline(atom_params[0], color='k', dashes=[3,3], lw=1.0, label='atom position')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend(loc=4,fontsize=8)
        plt.show()

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
    # #plt.imshow(np.abs(N)**2, aspect='auto', norm=LogNorm(vmin=0.1, vmax=100))
    # plt.imshow(np.abs(N)**2, aspect='auto')
    # plt.colorbar()
    # plt.show()
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

def linear_dispersion_scattering(k, N, T, atom_params, phase_zero_offset=None,
                                formula_option='Full'):
    """ Calculates the scattering matrix of a 1D cavity containing a single atom
        at a given position.
        Input format:
        - k: a 1D array/list of k-values or frequencies to compute the spectra on.
        - N: [N0, N1, N2, ..., NN, NN+1]; refractive indices of different layers,
                                           where N0,NN+1 is the space to the left/right,
                                           respectively and the rest are the layers.
        - D: [-1, D1, D2, ..., DN, -1];   thicknesses of the different layers,
                                           left/right space do not have thicknesses,
                                           the rest are the layers
        - atom_params: [atom_pos, atom_dPol, atom_om, atom_gamma];
            - atom_pos:   float;  indicates the position of the atom relative to the left
                                  boundary
            - atom_dPol:  float;  dipole moment of the atom in arbitrary units
            - atom_om:    float;  transition frequency of the atom in the same units as k
            - atom_gamma: float;  decay rate into *other* channels, same units as k
        - phase_zero_offset: None or array of length k;
                             An offset phase indicating where the zero of your
                             position coordinate system is.
        - formula_option: There are different versions of linear dispersion theory in the
                          lirature. This function allows for two of them.
            - 'Full': Full linear dispersion theory
                      (see e.g. https://doi.org/10.1103/PhysRevResearch.2.023396),
                      only neglecting the A^2 term
                      (see e.g. https://doi.org/10.1103/PhysRevA.93.012120).
            - 'Rot':  Standard version TODO

    """
    atom_pos, atom_dPol, atom_om, atom_gamma = atom_params

    # find the index of the layer the atom is in and the position of the start of that layer:
    t_cumul = 0.0
    for i,t in enumerate(T[1:-1]):
        t_cumul += t
        if t_cumul>atom_pos:
            t_cumul -= t
            ind_atomLayer = i+1
            n_atomLayer = N[i+1] # refractive index of the layer containing the atom
            break

    gamma_eff = atom_gamma

    t_min = min(np.amin(T[1:-1]), 1.0/np.amax(k))
    t_small = t_min/100000000.0 # -> simulates delta function
    num_dens = 1.0/t_small     # "number density" -> simulates one atom smeared over simulated delta function
    
    ###################################################################
    ### Linear dispersion theory formula ##############################
    ###################################################################
    if formula_option == 'Full':
        '''
        Version from my deriviation: does not assume the rotating-wave
        approximation. The A^2 term is however neglected, since this
        would involve an additional constant.
        '''
        susc_atom = -2.*atom_dPol**2 * num_dens*atom_om**3/(k**2-atom_om**2) \
                     /(k**2)
        n_atom = np.sqrt(1.0+0j + susc_atom)
    elif formula_option == 'Rot':
        '''
        Standard version of the linear dispersion formula, e.g. from
        lecture by J. Evers (Heidelberg) or from Roehlsberger 2004.
        There are TWO approximations implicit here:
            1. Rotating-wave approximation.
            2. atom_om**2/k**2 ~ 1
        Note that 2. already breaks in the multi-mode strong coupling
        regime, where 1. often still holds.
        Note also that n_atom = 1.0+0j + susc_atom/2.0
        does not hold here because susc is large due to the thin layer/
        single atom assumption.
        '''
        susc_atom = 2.0*atom_dPol**2 * num_dens \
                    * (-1.0)/(1j*gamma_eff + 2.0*(k-atom_om))
        n_atom = np.sqrt(1.0+0j + susc_atom)
        
    ###################################################################
    ### Assemble layer system #########################################
    ###################################################################

    N_int = [None] * (np.size(N)+2)
    T_int = [None] * (np.size(T)+2)
    for i, (n, t) in enumerate(zip(N_int, T_int)):
        if ind_atomLayer>i:
            N_int[i] = N[i]
            T_int[i] = T[i]
        elif ind_atomLayer+2<i:
            N_int[i] = N[i-2]
            T_int[i] = T[i-2]
        elif ind_atomLayer==i:
            N_int[i]   = N[i]
            N_int[i+2] = N[i]
            T_int[i]   = atom_pos-t_cumul-t_small/2.0  # first surrounding layer
            T_int[i+2] = T[i]-T_int[i]-t_small # second surrounding layer
            N_int[i+1] = n_atom
            T_int[i+1] = t_small               # atom layer

    return N_int, T_int, parratt_maxwell1D_matrix(N_int, T_int, k, phase_zero_offset=phase_zero_offset)

def linear_dispersion_scattering_multi_atom(k, N, T, atoms_params, phase_zero_offset=None,
                                formula_option='Full'):
    Nc = copy.deepcopy(N)
    Tc = copy.deepcopy(T)

    atoms_pos = []

    t_min = min(np.amin(T[1:-1]), 1.0/np.amax(k)) # not using Tc here to prevent smaller numbers for more atoms
    t_small = t_min/100000000.0 # -> simulates delta function

    # check of two atoms are close enough to be considered on top of each other; if so combine
    on_top = [None] * len(atoms_params)
    # False -> lonly atom; index list -> indices of other atoms are on top; True -> appears in index list of a previous atom
    for i, atom_params in enumerate(atoms_params):
        if on_top[i] is None:
            on_top_layer_idxs = []
            atom_pos_1 = atom_params[0]
            any_redunant = False
            for j, atom_params2 in enumerate(atoms_params):
                if j>i:
                    atom_pos_2 = atom_params2[0]
                    if np.abs(atom_pos_1-atom_pos_2)<4.*t_small:
                    # hardcoded threshold; all closer atoms are considered to be on top of each other
                        any_redunant = True
                        on_top_layer_idxs.append(j)
                        on_top[j] = True
            if any_redunant:
                on_top[i] = on_top_layer_idxs
            else:
                on_top[i] = False

    for a_i, atom_params in enumerate(atoms_params):
        if not on_top[a_i] == True:
            atom_pos, atom_dPol, atom_om, atom_gamma = atom_params
            atoms_pos.append(atom_pos)

            # find the index of the layer the atom is in and the position of the start of that layer:
            t_cumul = 0.0
            for i,t in enumerate(Tc[1:-1]):
                t_cumul += t
                if t_cumul>atom_pos:
                    t_cumul -= t
                    ind_atomLayer = i+1
                    n_atomLayer = Nc[i+1] # refractive index of the layer containing the atom
                    break

            gamma_eff = atom_gamma

            num_dens = 1.0/t_small     # "number density" -> simulates one atom smeared over simulated delta function
            
            ###################################################################
            ### Linear dispersion theory formula ##############################
            ###################################################################
            if formula_option == 'Full':
                '''
                Version from TODO: does not assume the rotating-wave
                approximation. The A^2 term is however neglected (see TODO).
                '''
                susc_atom = -2.*atom_dPol**2 * num_dens*atom_om**3/(k**2-atom_om**2) \
                             /(k**2) # TODO: include gamma_eff here?!
            elif formula_option == 'Rot. wave':
                '''
                Standard version of the linear dispersion formula, e.g. from
                TODO.
                There are TWO approximations implicit here:
                    1. Rotating-wave approximation.
                    2. atom_om**2/k**2 ~ 1
                Note that 2. already breaks in the multi-mode strong coupling
                regime, where 1. often still holds.
                Note also that n_atom = 1.0+0j + susc_atom/2.0
                does not hold here because susc is large due to the thin layer/
                single atom assumption.
                '''
                susc_atom = 2.0*atom_dPol**2 * num_dens \
                            * (-1.0)/(1j*gamma_eff + 2.0*(k-atom_om))

            if not (type(on_top[a_i]) == bool):
                for j in on_top[a_i]:
                    atom_pos_j, atom_dPol_j, atom_om_j, atom_gamma_j = atoms_params[j]
                    gamma_eff_j = atom_gamma_j
                    # add susceptibilities of atoms/transitions that are on top of each other.
                    if formula_option == 'Full':
                        susc_atom += -2.*atom_dPol_j**2 * num_dens*atom_om_j**3/(k**2-atom_om_j**2) \
                                     /(k**2) # TODO: include gamma_eff here?!
                    elif formula_option == 'Rot. wave':
                        susc_atom += 2.0*atom_dPol_j**2 * num_dens \
                                    * (-1.0)/(1j*gamma_eff_j + 2.0*(k-atom_om_j))

            n_atom = np.sqrt(1.0+0j + susc_atom)
                
            ###################################################################
            ### Assemble layer system #########################################
            ###################################################################

            N_int = [None] * (len(Nc)+2)
            T_int = [None] * (len(Tc)+2)
            for i, (n, t) in enumerate(zip(N_int, T_int)):
                if ind_atomLayer>i:
                    N_int[i] = Nc[i]
                    T_int[i] = Tc[i]
                elif ind_atomLayer+2<i:
                    N_int[i] = Nc[i-2]
                    T_int[i] = Tc[i-2]
                elif ind_atomLayer==i:
                    N_int[i]   = Nc[i]
                    N_int[i+2] = Nc[i]
                    T_int[i]   = atom_pos-t_cumul-t_small/2.0  # first surrounding layer
                    T_int[i+2] = Tc[i]-T_int[i]-t_small # second surrounding layer
                    N_int[i+1] = n_atom
                    T_int[i+1] = t_small 

            Nc = copy.deepcopy(N_int)
            Tc = copy.deepcopy(T_int)

    return N_int, T_int, parratt_maxwell1D_matrix(N_int, T_int, k, phase_zero_offset=phase_zero_offset)











