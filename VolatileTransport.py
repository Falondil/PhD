# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:00:06 2024

@author: Viktor Sparrman
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import erf 
import math
import time
import matplotlib as mpl

truncnormdist = True

volatiles = np.array(['hydrogen', 'carbon', 'nitrogen', 'oxygen'])
vcolors = np.array(['c', 'r', 'g', 'b'])
irange = range(len(volatiles))

pi = np.pi
sqrtpi = np.sqrt(pi)

# units: years, km, Pg (Petagram), Kelvin
rsurf = 6371 # km
rc = rsurf*0.3 # km
magmadepth = rsurf-rc # km
vmet = 1*365.242199*86400/1e3 # km yr-1, falling speed of metal blobs
silicatedensity = 3.4 # Pg km-3
pebblemetalfrac = 0.3 # mass fraction of metal (iron) in accreted pebbles
pebblerate = 430000 # Pg yr-1, PLACEHOLDER VALUE

number_of_boundaries = 100 # set the number of boundaries between layers of the magma ocean (also includes surface and core layer boundaries)
rk, dr = np.linspace(rc, rsurf, number_of_boundaries, retstep=True) # value of the spherical coordinate r at each layer boundary
delrk = np.array([rk[:-1]-r for r in rk[:-1]]) # matrix delrk[k, k'] with elements rk' - rk for all ordered pairs (rk', rk). 

# convection
Dmin = 0.2*365.242199*86400/1e6 # km2 yr-1, PLACEHOLDER value stolen from AnatomyI for Vesta values
Dmax = Dmin*1e3 # "2 to 3 orders of magnitude greater than for Vesta"
cspread = 1 # allow the 1 std diffusion of particles from one layer to maximally spread _ layers
stdmax = cspread*dr # km
dt = stdmax**2/(2*Dmax) # years, set the timestep long enough that std spread is at most _ layers
stdmin = np.sqrt(2*Dmin*dt) # km

number_of_layers = number_of_boundaries-1
layercenters = rk[:-1]+dr/2 # create array of distances from planetary center to center of each magma ocean sublayer
layervolumes = 4*pi*(rk[1:]**3-rk[:-1]**3) # calculate the volume of each layer
pressures = np.zeros(number_of_layers) # set the pressure in each layer
temperatures = np.zeros(number_of_layers) # set the temperature in each layer
fi0 = 0.1*np.array([1/2,1/3,1/4,1/5]) # initial mass fraction of volatiles, PLACEHOLDER
Pi0 = np.array([5, 300, 10, 100])
Pik0 = np.array([P0*np.ones_like(layercenters) for P0 in Pi0]) # nominal values, PLACEHOLDER. 

M0 = silicatedensity*layervolumes # mass of layer k, Pg PLACEHOLDER
vmasses = np.array([[f0*M for M in M0] for f0 in fi0]) # total mass of volatiles in each layer, Mik, [Pg]. PLACEHOLDER
Mmet = pebblerate*pebblemetalfrac*dr/vmet # constant mass [Pg] for the metal in any one layer resulting from isotropic accretion with constant falling speed. 
massfrac = Mmet/M0 # Metal silicate fraction in each magma ocean sublayer
vmassinmet = vmasses*(1+1/(massfrac*Pik0))**(-1)

#------------------------------------------------------------------------------
def massfromconvection(masses, sigmak, metsilfrac=0, Pik=0): # function for computing new masses (of any element) in each magma ocean sublayer resulting from convection after a timestep dt 
    konlyfactor = masses/2*(metsilfrac*Pik+1)**(-1)*sigmak/dr # compute factor that depends only on previous layer (index k)
    # compute the arguments for the function in the mass transfer equation
    x = delrk/sigmak
    xplus = (delrk+dr)/sigmak
    xminus = (delrk-dr)/sigmak
    kprimefactor = xplus*erf(xplus)-2*x*erf(x)+xminus*erf(xminus)+np.exp(-xplus**2)/sqrtpi-2*np.exp(-x**2)/sqrtpi+np.exp(-xminus**2)/sqrtpi # compute the factor depending on the new layer (index k')
    Mkprimek = konlyfactor*kprimefactor # matrix ([k, k'] format) of how mass from layer k has spread to layers k'
    if truncnormdist:
        Mkprimek *= masses/np.sum(Mkprimek, axis=1) # sum over all k' and upscale so that mass is conserved. FLAG HERE
    Mkprime = np.sum(Mkprimek, axis=0) # compute mass in new layer k' as sum of mass contribution from each other layer k. (Axis = 0 is the k index axis).
    if not truncnormdist:
        Mkprime += masses-np.sum(Mkprimek, axis=1) # add back the mass that was lost
    return Mkprime

starttime = time.time()

Msil = np.copy(M0)
for j in range(100):
    Msil = massfromconvection(Msil, stdmax, massfrac, 0)
    massfrac = Mmet/Msil
    vmasses = np.array([massfromconvection(vmasses[i], stdmax, massfrac, Pik0[i])+vmassinmet[i] for i in irange]) # PLACEHOLDER, the added massfrac*vmass term corresponds to metal deposition speed = 0. 
    vmassinmet = vmasses*(1+1/(massfrac*Pik0))**(-1)

    plt.figure()
    plt.plot(layercenters, M0, '-.', color='k', label='initial silicon mass')
    plt.plot(layercenters, Msil, '.', color='k', label='silicon mass after '+str(j+1)+' convection steps')
    plt.legend()
    plt.grid()
    plt.xlabel('Distance from planet center [km]')
    plt.ylabel('Mass [Pg]')
    plt.title('Silicon mass, Timestep number: '+str(j+1))
    
    # plt.figure()
    # ax = plt.gca()
    # ax.set_prop_cycle('color', vcolors)
    # plt.plot(layercenters, np.matrix.transpose(vmasses), '.',label=volatiles)
    # plt.legend()
    # plt.grid()
    # plt.xlabel('Distance from planet center [km]')
    # plt.ylabel('Mass [Pg]')
    # plt.title('Volatile masses, Timestep number: '+str(j+1))

endtime = time.time()
timeelapsed = endtime-starttime

#---------------------------------Legacy---------------------------------------
# # sampling for future approximation of erf(x)
# def A(x): # one part of upper limit for the difference |x-x'| 
#     return sqrtpi/2*erf(abs(x))*np.exp(x**2)

# def B(x): # other part of upper limit for the difference |x-x'|
#     return 1/(2*abs(x))

# def U(x): # upper limit function for the difference |x-x'|, MIGHT NOT BE NEEDED
#     Uelement = (A(x), B(x))
#     return np.min(Uelement, axis=0)

# def linearsampling(Stdmin=stdmin, Stdmax=stdmax, drmin=dr, drmax=magmadepth): # find at what values the error function has to be sampled at to have good coverage
#     stepsize = min(A(drmin/Stdmax), B(drmax/Stdmin))
#     xprimes = np.arange(drmin/Stdmax, drmax/Stdmin, stepsize)
#     return np.append(xprimes, xprimes[-1]+stepsize)

# def taylorerf(x, erfsample, exponentialsample, fractionsample): # Taylor expansion of error expansion
#     quit 

# number_of_terms = 6 # number of terms of the Taylor expansion to include
# xprime = linearsampling(drmax=5*dr) # PLACEHOLDER, sampling has to be smarter. 
# erfxprime = erf(xprime) # sampled values for erf(x')
# eminusxprimesquared = np.exp(-xprime**2) # sampled values for e^(-x'^2)
# jmatrix = np.array([[(-2*x)**j/math.factorial(j+1) for j in range(1, number_of_terms-1)] for x in xprime]) # calculates [(-2x')^1/(2!), (-2x')^2/(3!), ...] for all x'

# plt.figure() Plotting the upper bound function in the sampling domain
# plt.loglog(xprime, A(xprime),'.',color='r')
# plt.loglog(xprime, B(xprime), '.',color='b')
# plt.axis([xprime[0], xprime[-1], B(xprime[-1]), B(xprime[0])])