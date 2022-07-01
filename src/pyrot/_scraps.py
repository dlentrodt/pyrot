# def j_from_z(z, N, T): # z in [L]
#     """
#     Convert depth from cavity surface into layer index + depth from layer surface.

#     Returns the layer index j and depth from the layer boundary z-z_j
#     given the total depth z and a cavity.
#         - j=0 corresponds to the first layer (vacuum in pynuss) where z<0.
#           The distance to the upper layer boundary is not defined in this case
#           and given as -z (TODO: check that field formula applies in this region)
#         - j=1 is the first layer, with layer boundary position z_1 = 0
#         - j=2 is the second layer, with layer boundary position z_2 = t_1
#           (t_1: thickness of the first layer)
#         - j>2 is treated analogously.
#     """
#     Thicknesses = T[1:]
#     print(Thicknesses)
#     if z<0.:
#         return 0, z
#     if z==0.:
#         return 1, z
#     for j, t in enumerate(Thicknesses[0:-1]):
#         if ( sum(Thicknesses[0:j]) < z ) and ( sum(Thicknesses[0:j+1]) >= z):
#             return j+1, z-np.sum(Thicknesses[0:j])
#     return j+2, z-np.sum(Thicknesses[0:j+1]) # returns index and sum of layer thicknesses above in [L]

# def Εs_0(z, N, T, Theta, omega):
#     Field = np.zeros_like(z, dtype=np.complex128)
#     for i,zi in enumerate(z):
#         n = len(N)-1
#         j, z_offset = j_from_z(zi, N, T)
#         betaj = beta_j(j, N, T, Theta, omega)
#         dj = T[j-1] # [L]
#         if j==0 or j==(len(N)-1):
#             dj = 0.
#         rs_j0 = r_i_j(j, 0, N, T, Theta, omega, pol='s') # = rs_j-
#         rs_jn = r_i_j(j, n, N, T, Theta, omega, pol='s') # = rs_j+
#         ts_0j = t_i_j(0, j, N, T, Theta, omega, pol='s')
#         Dsj = 1. - rs_j0 * rs_jn * np.exp(2j*betaj*dj)
#         zm = z_offset
#         zp = dj - z_offset
#         Field[i] = ts_0j*np.exp(1j*betaj*dj)/Dsj * ( np.exp(-1j*betaj*zp) +  rs_jn*np.exp(+1j*betaj*zp) )
#     return z, Field

# def Εs_n(z, N, T, Theta, omega):
#     Field = np.zeros_like(z, dtype=np.complex128)
#     for i,zi in enumerate(z):
#         n = len(N)-1
#         j, z_offset = j_from_z(zi, N, T)
#         betaj = beta_j(j, N, T, Theta, omega)
#         dj = T[j-1] # [L]
#         if j==0 or j==(len(N)-1):
#             dj = 0.
#         rs_j0 = r_i_j(j, 0, N, T, Theta, omega, pol='s') # = rs_j-
#         rs_jn = r_i_j(j, n, N, T, Theta, omega, pol='s') # = rs_j+
#         ts_nj = t_i_j(n, j, N, T, Theta, omega, pol='s')
#         Dsj = 1. - rs_j0 * rs_jn * np.exp(2j*betaj*dj)
#         zm = z_offset
#         zp = dj - z_offset
#         Field[i] = ts_nj*np.exp(1j*betaj*dj)/Dsj * ( np.exp(-1j*betaj*zm) +  rs_j0*np.exp(+1j*betaj*zm) )
#     return z, Field

# def beta_j(j, N, T, Theta, omega):
#     ## omega = omega.TransitionEnergy # [keV] ## OLD # [keV]
#     k = omega # [1/L]
#     k_parallel = 0. #k*np.cos(Theta) # [1/L]
#     betaj = np.sqrt(N[j]**2*k**2-k_parallel**2)
#     return betaj # [1/L]

# def D_j_i_k(j, i, k, N, T, Theta, omega, pol='s'):
#     betaj = beta_j(j, N, T, Theta, omega)
#     dj = T[j] # [m]
#     rj_i = r_i_j(j, i, N, T, Theta, omega, pol=pol)
#     rj_k = r_i_j(j, k, N, T, Theta, omega, pol=pol)
#     return 1. - rj_i*rj_k*np.exp(2.j*betaj*dj)

# def gamma_ij(i, j, N, T, Theta, omega, pol='s'):
#     ### single interface, abs(i-j)=1 ###
#     if not (np.abs(i-j) == 1):
#         raise ValueError('Not adjacent layers, gamma_ij not defined.')
#     if pol=='s':
#         return 1.+0.j
#     ϵi = N[i]**2
#     ϵj = N[j]**2
#     return ϵi/ϵj

# def r_ij(i, j, N, T, Theta, omega, pol='s'):
#     if not (np.abs(i-j) == 1):
#         raise ValueError('Not adjacent layers, r_ij not defined.')
#     betai = beta_j(i, N, T, Theta, omega)
#     betaj = beta_j(j, N, T, Theta, omega)
#     gammaij = gamma_ij(i, j, N, T, Theta, omega, pol=pol)
#     return (betai - gammaij*betaj)/(betai + gammaij*betaj)

# def t_ij(i, j, N, T, Theta, omega, pol='s'):
#     if not (np.abs(i-j) == 1):
#         raise ValueError('Not adjacent layers, t_ij not defined.')
#     gammaij = gamma_ij(i, j, N, T, Theta, omega, pol=pol)
#     rij = r_ij(i, j, N, T, Theta, omega, pol=pol)
#     return np.sqrt(gammaij)*(1. + rij)

# def r_i_j_k(i, j, k, N, T, Theta, omega, pol='s'):
#     ### recurrence relation ###
#     betaj = beta_j(j, N, T, Theta, omega)
#     dj = T[j] # [m] TODO: units
#     Dj = D_j_i_k(j, i, k, N, T, Theta, omega, pol=pol)
#     ri_j = r_i_j(i, j, N, T, Theta, omega, pol=pol)
#     rj_i = r_i_j(j, i, N, T, Theta, omega, pol=pol)
#     rj_k = r_i_j(j, k, N, T, Theta, omega, pol=pol)
#     ti_j = t_i_j(i, j, N, T, Theta, omega, pol=pol)
#     tj_i = t_i_j(j, i, N, T, Theta, omega, pol=pol)
#     return 1./Dj * ( ri_j + (ti_j*tj_i - ri_j*rj_i) * rj_k * np.exp(2j*betaj*dj) )

# def t_i_j_k(i, j, k, N, T, Theta, omega, pol='s'):
#     ### recurrence relation ###
#     betaj = beta_j(j, N, T, Theta, omega)
#     dj = T[j] # [m]
#     Dj = D_j_i_k(j, i, k, N, T, Theta, omega, pol=pol)
#     ti_j = t_i_j(i, j, N, T, Theta, omega, pol=pol)
#     tj_k = t_i_j(j, k, N, T, Theta, omega, pol=pol)
#     return 1./Dj * ti_j*tj_k * np.exp(1j*betaj*dj)

# def r_i_j(i, j, N, T, Theta, omega, pol='s'):
#     ### starts and ends the recurrence chain ###
#     if np.abs(i-j) == 1:
#         return r_ij(i, j, N, T, Theta, omega, pol=pol)
#     if i==j:
#         return 0.+0.j
#     # choose middle index to start recurrence chain #
#     if i>j:
#         k=i-1
#     else:
#         k=i+1
#     return r_i_j_k(i, k, j, N, T, Theta, omega, pol=pol)

# def t_i_j(i, j, N, T, Theta, omega, pol='s'):
#     ### starts and ends the recurrence chain ###
#     if np.abs(i-j) == 1:
#         return t_ij(i, j, N, T, Theta, omega, pol=pol)
#     if i==j:
#         return 1.+0.j
#     # choose middle index to start recurrence chain #
#     if i>j:
#         k=i-1
#     else:
#         k=i+1
#     return t_i_j_k(i, k, j, N, T, Theta, omega, pol=pol)

# def GF(z, z0, N, T, Theta, omega):
#     """
#         This function only implements the s-polarisation part of the Green's function.
#         Since we are considering 1D cavities, however, this is the only relevant part.
#     """
#     xis = -1
#     n = len(N)-1
#     betan = beta_j(n, N, T, Theta, omega)
#     ts_0n = t_i_j(0, n, N, T, Theta, omega)
#     # only single pol (s):
#     zs,Es0_1 = Εs_0(z, N, T, Theta, omega)
#     zs,Esn_1 = Εs_n(z0, N, T, Theta, omega)
#     zs,Es0_2 = Εs_0(z0, N, T, Theta, omega)
#     zs,Esn_2 = Εs_n(z, N, T, Theta, omega)
#     Z0, Z = np.meshgrid(z0, z) # note reversed order for consistency with np.outer
#     heavi_1 = np.heaviside(np.real(Z-Z0), 0.5)
#     heavi_2 = np.heaviside(np.real(Z0-Z), 0.5)
#     return 2j*np.pi/betan * xis/ts_0n * ( np.outer(Es0_1, Esn_1)*heavi_1 + np.outer(Esn_2, Es0_2)*heavi_2 )