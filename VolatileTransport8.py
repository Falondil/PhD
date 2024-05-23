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

# choices
RBMdirk = True # decide to use analytical infinite series expression of diffusive motion from Dirk Veestraeten 2004. 
RBMnterms = 3 # decide how many positive and negative n terms of the infinite series expansion of 1/sinh to use in the PDF expression. 
renormdist = False # decide to use a truncated & renormalized normal distribution for the spread of mass
addtosource = False # decide if mass spread outside the inner and outer magma ocean boundaries should be re-added to the source layer
cspread = 3 # allow the 1 std diffusion of particles from one layer to maximally spread _ layers
plotstepspacing = 3 # Every n timesteps plots are printed  
number_of_compression_timesteps = 100

# initializing
volatiles = np.array(['hydrogen', 'carbon', 'nitrogen', 'oxygen'])
vcolors = np.array(['c', 'r', 'g', 'b'])
irange = range(len(volatiles))

pi = np.pi
sqrtpi = np.sqrt(np.pi)
sqrt2 = np.sqrt(2)

# units: years, km, Pg (Petagram)
simulatedtime = 1e3 # yr
rsurf = 6371 # km
rc = rsurf*0.3 # km
magmadepth = rsurf-rc # km
vmet = 1*365.242199*86400/1e3 # km yr-1, falling speed of metal blobs
silicatedensity = 3.4 # Pg km-3
metaldensity = 5.7 # Pg km-3, PLACEHOLDER
mc = 4/3*pi*rc**3*metaldensity # Pg, mass of core
pebblemetalfrac = 0.3 # mass fraction of metal (iron) in accreted pebbles
# pebblerate = 4.7 # Pg yr-1. Calculated from model description from Marie-Luise Steinmayer 2023: Mdotaccretion = 2/3^(2/3)*(St=0.01/0.1)^(2/3)*sqrt((GMstar)=7.11e-5 au^3/yr^2)*(xi=0.003)*(Mgdot=1e-9 Msun/year)/(2*pi*((chi=15/14+3/14+3/2)+(3/2)*(alpha=1e-3))*(cs_0=0.137 au year-1)^2)*((Mplanet/Msun)=0.6*3e-6)^(2/3). Paste into google: # 2/3^(2/3)*(0.01/0.1)^(2/3)*sqrt(7.11e-5)*(0.003)*(1e-9)/(2*pi*((15/14+3/14+3/2)+(3/2)*(1e-3))*(0.137)^2)*(0.6*3e-6)^(2/3)
pebblerate = 1e5 # Pg yr-1, PLACEHOLDER VALUE, one order of magnitude less than what is required to form a 1 Earth mass planet over 5 Myr.
psurf = 1e11 # Pg km-1 yr-2. PLACEHOLDER VALUE. 1e11 = 1 atm.
G = 66465320.9 # gravitational constant, km3 Pg-1 yr-2
GPaconversion = 1.0041778e-15 # conversion to GPa from Pg, km, years units

# compression and expansion values
alpha0 = 3e-5 # volumetric thermal expansion coefficient, K-1
K0 = 200 # bulk modulus, GPa
K0prime = 4 # bulk modulus derivative w.r.t pressure
psurf = 10/1e4 # surface pressure, GPa. corresponds to 10 bar. 
Tsurf = 300 # surface temperature, K, PLACEHOLDER
Cpsil = 1.19500749e12 # silicate melt heat capacity at constant pressure (1200 J kg-1 K-1), km2 yr-2 K-1 

number_of_boundaries = 100 # set the number of boundaries between layers of the magma ocean (also includes surface and core layer boundaries)
rk, dr = np.linspace(rc, rsurf, number_of_boundaries, retstep=True) # value of the spherical coordinate r at each layer boundary
delrk = np.array([rkprime-rk[:-1] for rkprime in rk[:-1]]) # matrix delrk[k', k] with elements rk' - rk. 

# convection
Dmin = 0.2*365.242199*86400/1e6 # km2 yr-1, PLACEHOLDER value stolen from AnatomyI for Vesta values
Dmax = Dmin*1e3 # "2 to 3 orders of magnitude greater than for Vesta"
stdmax = cspread*dr # km
dt = stdmax**2/(2*Dmax) # years, set the timestep long enough that std spread is at most _ layers
stdmin = np.sqrt(2*Dmin*dt) # km
number_of_timesteps = int(simulatedtime/dt) # compute the number of timesteps

number_of_layers = number_of_boundaries-1
layercenters = rk[:-1]+dr/2 # create array of distances from planetary center to center of each magma ocean sublayer
layervolumes = 4*pi*(rk[1:]**3-rk[:-1]**3) # calculate the volume of each layer
fi0 = 0.1*np.array([1/2,1/3,1/4,1/5]) # initial mass fraction of volatiles, PLACEHOLDER
Pi0 = np.array([5, 300, 10, 100]) # H, C, N, O
Pik0 = np.array([P0*np.ones_like(layercenters) for P0 in Pi0]) # nominal values, PLACEHOLDER. 

M0sil = silicatedensity*layervolumes # initial silicate mass of layer k, Pg PLACEHOLDER
Mmet = pebblerate*pebblemetalfrac*dr/vmet # constant mass [Pg] for the metal in any one layer resulting from isotropic accretion with constant falling speed. 
massfrac = Mmet/M0sil # Metal silicate fraction in each magma ocean sublayer

# option 1: even concentration of volatiles
# Mvol = np.array([[f0*M/(1-f0) for M in M0sil+Mmet] for f0 in fi0]) # total mass of volatiles in each layer, Mik, [Pg]. PLACEHOLDER
# option 2: delta function initial dist. of volatiles
Mvol = np.zeros((len(volatiles), number_of_layers))
Mvol[:, 0] = 1e10*fi0 # volatiles only exist in innermost layer

vmassinmet = Mvol*(1+1/(massfrac*Pik0))**(-1)

#------------------------------Structure functions-----------------------------
def delta(array): # calculates the forward difference to the next element in array. returns array with one less element than input array
    return array[1:]-array[:-1]    

def centeraveraging(array): # calculates the mean value between subsequent elements in array. returns array with one less element than input array
    return (array[1:]+array[:-1])/2

def totaldensity(Mtotk, Vk):
    return Mtotk/Vk

def menc(masses): # calculates mass enclosed below the lower boundary of each magma ocean layer
    return np.cumsum(np.concatenate(([0], masses[:-1])))+mc 

def thermalexpansion(pressure): # calculate the volumetric thermal expansion coefficient as a function of pressure, K-1
    return alpha0*(pressure*K0prime/K0+1)**((1-K0prime)/K0prime)

def bulkmodulus(pressure): # calculate the bulk modulus as a function of pressure
    return K0+pressure*K0prime

def pressurefunc(rvar, drvar, density, Menc):
    rflip = np.flip(rvar[:-1]) # flip the array of layerboundaries so it is ordered from the top down. r = rsurf excluded so last element cut before flip. 
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc) # also flip the enclosed mass array
    densityflip = np.flip(centeraveraging(density))
    return np.append(np.flip(psurf+GPaconversion*G*np.cumsum(densityflip*(2*pi/3*densityflip*(2*rflip*drflip+drflip**2)+(Mencflip-4*pi/3*rflip**3*densityflip)*(1/rflip-1/(rflip+drflip))))), psurf)

def pressurefunc2(rvar, drvar, density, Menc):
    rflip = np.flip(rvar[:-1]) # flip the array of layerboundaries so it is ordered from the top down. r = rsurf excluded so last element cut before flip. 
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc) # also flip the enclosed mass array
    densityflip = np.flip(centeraveraging(density))
    return np.append(np.flip(psurf+GPaconversion*G*np.cumsum(Mencflip*densityflip*drflip/rflip**2)), psurf)

def temperaturefunc(rvar, drvar, pressure, Menc): # calculate the temperature as a function of pressure and enclosed mass 
    rflip = np.flip(rvar[1:]) # flip the array of layerboundaries so it is ordered from the top down. r=rc excluded so first element cut before flip.
    drflip = np.flip(drvar) 
    Mencflip = np.flip(Menc) # also flip the enclosed mass array
    pressureflip = np.flip(centeraveraging(pressure)) # and the pressure array (innermost pressure is not needed)
    alphaflip = thermalexpansion(pressureflip) # calculate the flipped thermal expansion coefficient
    exponent = np.cumsum(G*alphaflip*Mencflip*drflip/(Cpsil*rflip**2)) # calculate the integral as a finite sum over each layer down to the layerboundary considered for the temperature
    return np.append(Tsurf*np.flip(np.exp(exponent)), Tsurf) # return the temperature at each layerboundary (including the surface)

def densityfunc(pressure, temperature): # calculate the new density as a function of the bulk compression and thermal expansion
    pressureflip = np.flip(pressure) # flip the pressure
    temperatureflip = np.flip(temperature) # flip the temperature
    alphaflip = centeraveraging(thermalexpansion(temperatureflip)) # calculate the thermal expansion coefficient assumed to be constant in each layer
    Kflip = centeraveraging(bulkmodulus(pressureflip)) # calculate the bulk modulus assumed to be constant in each layer 
    
    densityflip = np.zeros(number_of_boundaries)
    densityflip[0] = silicatedensity # assume silicatedensity at top layer
    for i in range(1, number_of_boundaries):
        densityflip[i] = densityflip[i-1]*np.exp((pressureflip[i]-pressureflip[i-1])/Kflip[i-1]-alphaflip[i-1]*(temperatureflip[i]-temperatureflip[i-1]))
    return np.flip(densityflip)

def newlayerboundaries(layermass, newdensity): # calculate the new boundaries for each layer given their new densities and old (conserved) masses
    rknew = np.zeros(number_of_boundaries)
    rknew[0] = rc
    for i in range(1, number_of_boundaries):
        rknew[i] = (layermass[i-1]/(4*pi*newdensity[i-1])+rknew[i-1]**3)**(1/3)
    return rknew

# def massanddensityinterpolation(rvar, newdensity): # interpolate the density from the new layer centers to the linearly spaced layer centers
#     rvarcenter = centeraveraging(rvar)
#     density = np.interp(rk+dr/2, rvar, newdensity)
#     layermass = density*layervolumes
#     densityatboundaries = np.interp(rk, rvar, newdensity)

#-----------------------------Parametrized functions---------------------------
def PCk(p, T, deltaIW = -2.2, Xsulfur = 0, Xoxygen = 0, NBOperT = 2.6, eq2=False): # Carbon partition coefficient. Parametrization from Fischer2020. Pressure, temperature in GPa, K units.
    if eq2:
        log10PCk = 1.81+2470/T-227*p/T+9.7*np.log10(1-Xsulfur)-30.6*np.log10(1-Xoxygen)-0.123*NBOperT-0.211*deltaIW
    else:
        log10PCk = 1.49+3000/T-235*p/T+9.6*np.log10(1-Xsulfur)-19.5*np.log10(1-Xoxygen)-0.118*NBOperT-0.238*deltaIW
    return 10**log10PCk

def PNk(p, T, XFeO,  Xsulfur=0, Xsilicon=0, NBOperT = 2.6): # Nitrogen partition function. Parametrization from Grewal2019. Pressure, temperature in GPa, K units.
    a, b, c, d, e, f, g, h = -1415.40, 5816.24, 166.14, 343.44, -38.36, 139.52, 0.82, 1.13
    log10PNk = a+b/T+c*p/T+d*np.log(1-Xsulfur)+e*np.log(1-Xsulfur)**2+f*np.log(1-Xsilicon)+g*NBOperT+h*np.log(XFeO) # calculate log10 of the partition coefficient for nitrogen.
    return 10**log10PNk # return the partition coefficient
#------------------------------Transport functions-----------------------------

def massfromconvection(masses, sigmak, metsilfrac=0, Pik=0): # function for computing new masses (of any element) in each magma ocean sublayer resulting from convection after a timestep dt 
    konlyfactor = masses/2*(metsilfrac*Pik+1)**(-1)*sigmak*sqrt2/dr # compute factor that depends only on previous layer (index k)
    # compute the arguments for the function in the mass transfer equation
    x = delrk/(sigmak*sqrt2)
    xplus = (delrk+dr)/(sigmak*sqrt2)
    xminus = (delrk-dr)/(sigmak*sqrt2)
    if RBMdirk:
        drc = rk[:-1]-rc # radial distance from core
        kprimefactor = np.zeros_like(x) # set = 0 before adding sequence of terms for different n
        for n in np.arange(-RBMnterms, RBMnterms+1):
            nrd = n*magmadepth # compute n*(r_OB-r_IB)
            x += 2*nrd # add n term to argument
            xplus += 2*nrd # add n term to argument
            xminus += 2*nrd # add n term to argument
            
            # create arguments for the extra terms added by considering RBM
            x2plusplus = (delrk+2*dr+2*nrd+2*drc)/(sigmak*sqrt2)           
            x2plus = (delrk+dr+2*nrd+2*drc)/(sigmak*sqrt2) 
            x2 = (delrk+2*nrd+2*drc)/(sigmak*sqrt2) 
            kprimefactor += xplus*erf(xplus)-2*x*erf(x)+xminus*erf(xminus)+np.exp(-xplus**2)*sqrt2/sqrtpi-2*np.exp(-x**2)*sqrt2/sqrtpi+np.exp(-xminus**2)*sqrt2/sqrtpi\
                +x2plusplus*erf(x2plusplus)-2*x2plus*erf(x2plus)+x2*erf(x2)+np.exp(-x2plusplus**2)*sqrt2/sqrtpi-2*np.exp(-x2plus**2)*sqrt2/sqrtpi+np.exp(-x2**2)*sqrt2/sqrtpi # Add the nth term in computing the factor depending on the new layer (index k')
    else:
        kprimefactor = xplus*erf(xplus)-2*x*erf(x)+xminus*erf(xminus)+np.exp(-xplus**2)*sqrt2/sqrtpi-2*np.exp(-x**2)*sqrt2/sqrtpi+np.exp(-xminus**2)*sqrt2/sqrtpi # compute the factor depending on the new layer (index k')
    Mkprimek = konlyfactor*kprimefactor # matrix ([k', k] format) of how mass from layer k has spread to layers k'
    if renormdist:
        Mkprimek *= np.divide(masses*(metsilfrac*Pik+1)**(-1), np.sum(Mkprimek, axis=0), out=np.zeros_like(masses), where=np.sum(Mkprimek, axis=0)!=0) # sum over all k' and upscale so that mass is conserved. FLAG HERE
    Mkprime = np.sum(Mkprimek, axis=1) # compute mass in new layer k' as sum of mass contribution from each other layer k. (Axis = 0 is the k index axis).
    if addtosource:
        Mkprime += masses*(metsilfrac*Pik+1)**(-1)-np.sum(Mkprimek, axis=0) # add back the mass that was lost. Compare the source mass to the spread of the source mass and re-add the lost mass.
    return Mkprime

def masschangefromsedimentation(masses, vmasses=Mvol, Pik=np.zeros(number_of_layers)): # function for computing the movement of volatiles between layers as resulting from descending metal blobs.
    stepnumber = vmet*dt/dr # calculate the number of layers that metal blobs descend through during one convective timestep dt. 
    massdifference = vmasses[1:]*Pik[1:]*Mmet/(masses[1:]+(Pik[1:]-1)*Mmet) - vmasses[:-1]*Pik[:-1]*Mmet/(masses[:-1]+(Pik[:-1]-1)*Mmet) # calculate descending volatile mass transfer from layers after the metal blobs descend 1 layerwidth.
    outermassdifference = -vmasses[-1]*Pik[-1]*Mmet/(masses[-1]+(Pik[-1]-1)*Mmet) # calculate the volatile mass loss from the outermost layer
    return stepnumber*np.append(massdifference, outermassdifference) # return the change in volatile mass in each layer as the result of sedimentation descent


# initialization
starttime = time.time()

Msil = np.copy(M0sil)
massfrac = Mmet/Msil
Mtot = np.copy(M0sil)+Mmet+np.sum(Mvol, axis=0)
Menc = menc(Mtot)

# replace initialpressure with proper calc from initial density of total mass not only silicate melt mass
# initialdensity = totaldensity(Mtot, layervolumes) # calculate the density of each layer
initialdensity = np.repeat(silicatedensity, number_of_boundaries) # initial density is assumed to be constant
initialpressure = psurf + GPaconversion*2*pi/3*silicatedensity**2*G*(rsurf**2-rk**2) # SOMETHING IS WRONG HERE. initial pressure in the magma ocean layerboundaries resulting from constant density profile, GPa
initialtemperature = temperaturefunc(rk, delta(rk), initialpressure, Menc)  # calculate the initial temperature in each layerboundary, unit K

for j in range(number_of_compression_timesteps):
    quit 
    
compressioncalctime = time.time()-starttime # calculate how much time in seconds is spent on calculating the initial differentiation of the planet

# Transport of volatiles loop

std = stdmax # Alt. 0: CONSTANT standard deviation of diffusion after a timestep dt for a point mass 
# std = stdmax-stdmax/2*np.sin(pi*(rk[:-1]-rc)/(rsurf-rc)) # Alt. 1. PLACEHOLDER sinusoidal dip from constant standard deviation
for j in range(number_of_timesteps):
    Mvolcopy = np.copy(Mvol) # copy how much volatile mass existed in previous timestep
    Mvol = np.array([massfromconvection(Mvol[i], std, massfrac, Pik0[i])+vmassinmet[i]+masschangefromsedimentation(Mtot, Mvol[i], Pik0[i]) for i in irange]) # PLACEHOLDER, the added massfrac*vmass term corresponds to metal deposition speed = 0. 
    massloss = np.sum(np.array([masschangefromsedimentation(Mtot, Mvol[i], Pik0[i]) for i in irange]))
    massdiff = np.sum(Mvolcopy)-np.sum(Mvol)
    
    vmassinmet = Mvol*(1+1/(massfrac*Pik0))**(-1)
    Mtot = Mmet+Msil+np.sum(Mvol, axis=0) # total mass in each layer

    if j%plotstepspacing==0: 
        # Use for single plotting
        # fig1 = plt.figure()
        # ax1 = plt.gca()
        # ax1.set_title('Mass in magma ocean layers after '+str(j+1)+' convection time steps')
        # fig2 = plt.figure()
        # ax2 = plt.gca()
        # ax2.set_title('Mass in magma ocean layers after '+str(j+1)+' convection time steps')
        
        # Use for subplots
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        # Plot masses
        Mvolsum = np.sum(Mvol)
        fig.suptitle('Mass in magma ocean layers after '+"{:.1f}".format((j+1)*dt)+' years\nTotal volatile mass = '+str(np.round(Mvolsum))+' Pg', y = 0.80)
        ax1.set(xlabel='Distance from planet center [km]', ylabel=('Silicate mass [Pg]'), aspect=3/4*(rsurf-rc)/max(np.concatenate((Msil,M0sil))))
        ax1.plot(layercenters, M0sil, '-.', color='k', label='initial silicate mass')
        ax1.plot(layercenters, Msil, '.', color='k', label='silicate mass')
        ax1.set_ylim(bottom=0)
        ax1.legend()
        ax1.grid()
        
        ax2.set(xlabel='Distance from planet center [km]', ylabel=('Volatile mass [Pg]'), aspect=3/4*(rsurf-rc)/np.max(Mvol))
        ax2.set_prop_cycle('color', vcolors)
        ax2.plot(layercenters, np.matrix.transpose(Mvol), '.',label=volatiles)
        ax2.set_ylim(bottom=0)
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_ticks_position('right')
        ax2.legend()
        ax2.grid()
        
        # # Plot densities
        # fig.suptitle('Density in magma ocean layers after '+"{:.1f}".format((j+1)*dt)+' years', y=0.85)
        # ax1.set(xlabel='Distance from planet center [km]', ylabel=('Silicate density [Pg km-3]'), aspect=3/4*(rsurf-rc)/max(Msil/layervolumes))
        # ax1.plot(layercenters, M0sil/layervolumes, '-.', color='k', label='initial silicate density')
        # ax1.plot(layercenters, Msil/layervolumes, '.', color='k', label='silicate density')
        # ax1.set_ylim(bottom=0)
        # ax1.legend()
        # ax1.grid()
        
        # ax2.set(xlabel='Distance from planet center [km]', ylabel=('Volatile density [Pg km-3]'), aspect=3/4*(rsurf-rc)/np.max(Mvol/layervolumes))
        # ax2.set_prop_cycle('color', vcolors)
        # ax2.plot(layercenters, np.matrix.transpose(Mvol/layervolumes), '.',label=volatiles)
        # ax2.set_ylim(bottom=0)
        # ax2.yaxis.set_label_position('right')
        # ax2.yaxis.set_ticks_position('right')
        # ax2.legend()
        # ax2.grid()
    print('Timestep '+str(j+1)+'/'+str(number_of_timesteps)+' completed')
    
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