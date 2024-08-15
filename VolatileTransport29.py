# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:00:06 2024

@author: Viktor Sparrman
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import math
import time

# choices
simulatedtime = 1e5  # yr

# decide to use analytical infinite series expression of diffusive motion from Dirk Veestraeten 2004.
RBMdirk = True
# decide how many positive and negative n terms of the infinite series expansion of 1/sinh to use in the PDF expression.
RBMnterms = 10
relativeconvection = False  # use Veestraeten2004 solution to RBM
explicit = not relativeconvection  # decide if explicit diffusion scheme is used
evendist = True  # decide initial distribution of volatiles to have a constant mass concentration throughout magma ocean
# decide initial distribution of volatiles to be concentrated in the layer closest to the core
innerdelta = not evendist

constantdensity = False  # decide to use the uncompressed silicate density throughout
# decide to use a truncated & renormalized normal distribution for the spread of mass
renormdist = False
addtosource = False  # decide if mass spread outside the inner and outer magma ocean boundaries should be re-added to the source layer
# decide to rescale the derived mass from convection transport equation by r^2 to turn 1D solution to 3D.
r2factor = True

cspread = 1  # allow the 1 std diffusion of particles from one layer to maximally spread _ layers
if explicit:
    cspread = 1  # can't have too much spread if diffusion only transports to nearest layer
plotstepspacing = 1  # Every n timesteps plots are printed
# plot at most ___ plots. Redefines plotstepspacing if necessary
max_number_of_plots = 200

fixedplanetmass = True  # incompatible with fixedplanetradius
# should the iterative steps to find the internal structure of the planet be plotted?
compressionplotting = True
number_of_structure_timesteps = 10
structureplotstepspacing = 1  # number_of_structure_timesteps

# initializing
volatiles = np.array(['hydrogen', 'carbon', 'nitrogen', 'oxygen'])
vcolors = np.array(['c', 'r', 'g', 'b'])
irange = range(len(volatiles))

pi = np.pi
sqrtpi = np.sqrt(np.pi)
sqrt2 = np.sqrt(2)

# conversions
GPaconversion = 1.0041778e-15  # conversion to GPa from Pg, km, year units
densityconversion = 1e3  # converts from Pg km-3 to kg m-3
secondsperyear = 31556926  # 31 556 926 seconds per year

# units: years, km, Pg (Petagram)
rc = 3480 # km
coreradiusfraction = 3480/6371  # Earth's Core chapter by David Loper. PLACEHOLDER
vmet = 1*secondsperyear/1e3  # 1 m/s in km yr-1, falling speed of metal blobs
silicatedensity = 3.4 # Pg km-3
# 5.7 # Pg km-3, Earth's Core chapter by David Loper. PLACEHOLDER
coremetaldensity = 10.48
mc = 4/3*pi*rc**3*coremetaldensity  # Pg, mass of core
earthmass = 5.9712e12  # Earth's mass in Pg
# pebblerate = 4.7 # Pg yr-1. Calculated from model description from Marie-Luise Steinmayer 2023: Mdotaccretion = 2/3^(2/3)*(St=0.01/0.1)^(2/3)*sqrt((GMstar)=7.11e-5 au^3/yr^2)*(xi=0.003)*(Mgdot=1e-9 Msun/year)/(2*pi*((chi=15/14+3/14+3/2)+(3/2)*(alpha=1e-3))*(cs_0=0.137 au year-1)^2)*((Mplanet/Msun)=0.6*3e-6)^(2/3). Paste into google: # 2/3^(2/3)*(0.01/0.1)^(2/3)*sqrt(7.11e-5)*(0.003)*(1e-9)/(2*pi*((15/14+3/14+3/2)+(3/2)*(1e-3))*(0.137)^2)*(0.6*3e-6)^(2/3)
pebblerate = 1*earthmass/5e6 # Pg yr-1, PLACEHOLDER VALUE, x order(s) of magnitude less than what is required to form a 1 Earth mass planet over 5 Myr.
pebblemetalfrac = 0.3  # mass fraction of metal (iron) in accreted pebbles

G = 66465320.9  # gravitational constant, km3 Pg-1 yr-2
Boltzmann = 1781.83355 # Boltzmann constant sigma = 5.67*1e-8 W/m^2 K^-4 converted to Petagram year^-3 K^-4

# Alternative. Not bad but perhaps insufficient as a guarantee so better not unless it works
if fixedplanetmass:
    planetscaling = 1
    # Earth was less massive before Theia giant impact. Perhaps core mass should be lower too pre-giant impact.
    protomass = planetscaling*earthmass
    rs0 = ((protomass-mc)/(4*pi*silicatedensity/3)+rc**3)**(1/3)
else:
    rs0 = 6371  # km

# compression and expansion values
alpha0 = 3e-5  # volumetric thermal expansion coefficient, K-1
# base heat conductivity in AnatomyI, 4 W m-1 K-1 in Pg km year-3 K-1
heatconductivity = 4*31425635.8
K0 = 200  # bulk modulus, GPa
K0prime = 4  # bulk modulus derivative w.r.t pressure
Mearthatm = 5.15e6  # Pg, current mass of Earth's atmosphere
atmospherescaling = 93 # how much thicker atmosphere than current Earth
Matm = atmospherescaling*Mearthatm*planetscaling  # Pg, mass of the atmosphere of the protoplanet modelled
psurf = 10/1e4  # surface pressure, GPa. corresponds to 10 bar. PLACEHOLDER
opacity = 1e4 # km2 Pg-1, PLACEHOLDER
Tsurf = 2400  # surface temperature, K. 1600 K from Tackley2012, 3200 K from Monteux2016 PLACEHOLDER
deltaT = 1000  # temperature above that of an adiabat at the core with same surface temperature, K, PLACEHOLDER
# converts specific heat capacity from units of J kg-1 K-1 to km2 yr-2 K-1
Cpconv = 995839577
# silicate melt heat capacity at constant pressure (1200 J kg-1 K-1), km2 yr-2 K-1
Cpsil = Cpconv*1200

number_of_structure_boundaries = int(1e4)
# value of the spherical coordinate r at each layer boundary
rs, drs = np.linspace(rc, rs0, number_of_structure_boundaries, retstep=True)

# ------------------------------Structure functions-----------------------------

def delta(array):  # calculates the forward difference to the next element in array. returns array with one less element than input array
    return array[1:]-array[:-1]


# calculates the mean value between subsequent elements in array. returns array with one less element than input array
def centeraveraging(array):
    return (array[1:]+array[:-1])/2


# calculates the centerpoint density between density calculated at boundaries.
def centerdensity(densityatboundaries, r):
    rcenters = r[:-1]+delta(r)/2
    # interpolate the shell density then divide by r^2 to get density.
    return np.interp(rcenters, r, r**2*densityatboundaries)/rcenters**2


def menc(masses):  # calculates mass enclosed below each layer boundary
    return np.cumsum(np.concatenate(([0], masses)))+mc


def gfunc(r, Menc):  # calculates local gravity at layer boundaries
    return G*Menc/r**2


# K-1, calculate the volumetric thermal expansion coefficient as a function in units of GPa of pressure. Abe1997
def thermalexpansion(pressure):
    return alpha0*(pressure*K0prime/K0+1)**((1-K0prime)/K0prime)


def bulkmodulus(pressure):  # calculate the bulk modulus as a function of pressure
    return K0+pressure*K0prime


def Cpfunc(temperature, material='e'):  # Shomate equation for specific heat capacity of material as function of temperature. NIST Webbook, Condensed phase thermochemistry data
    Tper1000 = temperature/1000  # Temperature/(1000 K)
    if material == 'e':  # enstatite, MgSiO3
        molecularweight = 100.3887/1000  # kg/mol
        a1, b1, c1, d1, e1 = 146.4400, -1.499926e-7, 6.220145e-8, - \
            8.733222e-9, -3.144171e-8  # 1850 K < T
        a2, b2, c2, d2, e2 = 37.72742, 110.1852, - \
            50.79836, 7.968637, 15.16081  # 903 K < T < 1850 K
        Cpermole = np.where(temperature > 1850, a1+b1*Tper1000+c1*Tper1000**2+d1*Tper1000**3+e1/Tper1000 **
                            2, a2+b2*Tper1000+c2*Tper1000**2+d2*Tper1000**3+e2/Tper1000**2)  # Shomate equation, J/(mol*K))
    elif material == 'f':  # forsterite, Mg2Si04
        molecularweight = 140.6931/1000  # kg/mol
        a1, b1, c1, d1, e1 = 205.0164, -2.196956e-7, 6.630888e-8, - \
            6.834356e-9, -1.298099e-7  # 2171 K < T
        a2, b2, c2, d2, e2 = 140.8749, 47.15201, - \
            12.22770, 1.721771, -3.147210  # T < 2171 K
        Cpermole = np.where(temperature > 2171, a1+b1*Tper1000+c1*Tper1000**2+d1*Tper1000**3+e1/Tper1000 **
                            2, a2+b2*Tper1000+c2*Tper1000**2+d2*Tper1000**3+e2/Tper1000**2)  # Shomate equation, J/(mol*K))
    else:
        sys.exit('No silicate material matches input')
    return Cpermole/molecularweight*Cpconv  # units of km2 year-2 K-1


def liquidus(pressure):  # calculate the liquidus temperature as a function of pressure in high and low pressure regime
    def Tliquidus(T0, p0, q):  # calculate the liquidus temperature as a function of pressure
        return T0*(1+pressure/p0)**q  # ref Monteux 2016 or AnatomyI.
    # use low pressure function parameters if pressure is low and vice versa
    return np.where(pressure < 20, Tliquidus(1982.2, 6.594, 0.186), Tliquidus(2006.8, 34.65, 0.542))


def solidus(pressure):  # calculate the liquidus temperature as a function of pressure in high and low pressure regime
    def Tsolidus(T0, p0, q):  # calculate the liquidus temperature as a function of pressure
        return T0*(1+pressure/p0)**q  # ref Monteux 2016 or AnatomyI.
    # use low pressure function parameters if pressure is low and vice versa
    return np.where(pressure < 20, Tsolidus(1661.2, 1.336, 1/7.437), Tsolidus(2081, 101.69, 1/1.226))


# calculates the pressure by integrating the density*gravitation down from the surface
def pressurefunc(rvar, drvar, density, Menc):
    # flip the array of layerboundaries so it is ordered from the top down. r = rs excluded so last element cut before flip.
    rflip = np.flip(rvar[:-1])
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc[:-1])  # also flip the enclosed mass array
    densityflip = np.flip(centerdensity(density, rvar))
    return np.append(np.flip(psurf+GPaconversion*G*np.cumsum(densityflip*(2*pi/3*densityflip*(2*rflip*drflip+drflip**2)+(Mencflip-4*pi/3*rflip**3*densityflip)*(1/rflip-1/(rflip+drflip))))), psurf)


def pressurefunc2(rvar, drvar, density, Menc):
    # flip the array of layerboundaries so it is ordered from the top down. r = rs excluded so last element cut before flip.
    rflip = np.flip(rvar[:-1])
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc[:-1])  # also flip the enclosed mass array
    densityflip = np.flip(centerdensity(density, rvar))
    return np.append(np.flip(psurf+GPaconversion*G*np.cumsum(Mencflip*densityflip*drflip/rflip**2)), psurf)


# calculate the temperature as a function of pressure and enclosed mass
def temperaturefunc(rvar, drvar, pressure, Menc):
    # flip the array of layerboundaries so it is ordered from the top down. r=rc excluded so first element cut before flip.
    rflip = np.flip(rvar[1:])
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc[1:])  # also flip the enclosed mass array
    # and the pressure array (innermost pressure is not needed)
    pressureflip = np.flip(centeraveraging(pressure))
    # calculate the flipped thermal expansion coefficient
    alphaflip = thermalexpansion(pressureflip)
    # calculate the integral as a finite sum over each layer down to the layerboundary considered for the temperature
    exponent = np.cumsum(G*alphaflip*Mencflip*drflip/(Cpsil*rflip**2))
    # return the temperature at each layerboundary (including the surface)
    return np.append(Tsurf*np.flip(np.exp(exponent)), Tsurf)+deltaT


# Layer step-wise calculating the temperature as a function of pressure and enclosed mass.
def temperaturefunc2(rvar, drvar, pressure, Menc):
    # flip the array of layerboundaries so it is ordered from the top down. r=rc excluded so first element cut before flip.
    rflip = np.flip(rvar[1:])
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc[1:])  # also flip the enclosed mass array
    # and the pressure array (innermost pressure is not needed)
    pressureflip = np.flip(centeraveraging(pressure))
    # calculate the flipped thermal expansion coefficient
    alphaflip = thermalexpansion(pressureflip)
    temperatureflip = np.zeros(number_of_structure_boundaries)
    # surface temperature for the first temperature of the adiabat
    temperatureflip[0] = Tsurf
    for i in range(1, number_of_structure_boundaries):
        temperatureflip[i] = temperatureflip[i-1] * \
            np.exp(drflip[i-1]*alphaflip[i-1]*G *
                   Mencflip[i-1]/(Cpsil*rflip[i-1]**2))
    # return the temperature at each layerboundary (including the surface)
    return np.flip(temperatureflip)+deltaT

# FIX HERE, CHOOSE ANOTHER TEMP. PROFILE AND IMPLEMENT


# Layer step-wise calculating the temperature as a function of pressure and enclosed mass using a variable specific heat capacity
def temperaturefunc3(rvar, drvar, pressure, Menc):
    # flip the array of layerboundaries so it is ordered from the top down. r=rc excluded so first element cut before flip.
    rflip = np.flip(rvar[1:])
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc[1:])  # also flip the enclosed mass array
    # and the pressure array (innermost pressure is not needed)
    pressureflip = np.flip(centeraveraging(pressure))
    # calculate the flipped thermal expansion coefficient
    alphaflip = thermalexpansion(pressureflip)
    # create array for storing temperatures of the adiabatic temperature profile
    adiabattemperatureflip = np.zeros(number_of_structure_boundaries)
    # surface temperature for the first temperature of the adiabat
    adiabattemperatureflip[0] = Tsurf
    # create array for storing temperatures of the adiabatic temperature profile
    temperatureflip = np.zeros(number_of_structure_boundaries)
    # surface temperature for the first temperature of the adiabat
    temperatureflip[0] = Tsurf
    for i in range(1, number_of_structure_boundaries):
        # set the specific heat capacity to dependent on the temperature of the layer above
        Cpflipadiabat = Cpfunc(adiabattemperatureflip[i-1])
        # set the specific heat capacity to dependent on the temperature of the layer above
        Cpflip = Cpfunc(temperatureflip[i-1])
        adiabattemperatureflip[i] = adiabattemperatureflip[i-1]*np.exp(
            drflip[i-1]*alphaflip[i-1]*G*Mencflip[i-1]/(Cpflipadiabat*rflip[i-1]**2))
        temperatureflip[i] = temperatureflip[i-1] * \
            np.exp(drflip[i-1]*alphaflip[i-1]*G *
                   Mencflip[i-1]/(Cpflip*rflip[i-1]**2))
    adiabattemperature = np.flip(adiabattemperatureflip)
    temperature = np.flip(temperatureflip)
    # return the temperature at each layerboundary (including the surface)
    return temperature, adiabattemperature


# Layer step-wise calculating the temperature as a function of pressure and enclosed mass using a variable specific heat capacity
def temperaturefunctilt(rvar, drvar, pressure, Menc):
    # flip the array of layerboundaries so it is ordered from the top down. r=rc excluded so first element cut before flip.
    rflip = np.flip(rvar[1:])
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc[1:])  # also flip the enclosed mass array
    # and the pressure array (innermost pressure is not needed)
    pressureflip = np.flip(centeraveraging(pressure))
    # calculate the flipped thermal expansion coefficient
    alphaflip = thermalexpansion(pressureflip)
    # create array for storing temperatures of the adiabatic temperature profile
    adiabattemperatureflip = np.zeros(number_of_structure_boundaries)
    # surface temperature for the first temperature of the adiabat
    adiabattemperatureflip[0] = Tsurf
    for i in range(1, number_of_structure_boundaries):
        # set the specific heat capacity to dependent on the temperature of the layer above
        Cpflip = Cpfunc(adiabattemperatureflip[i-1])
        adiabattemperatureflip[i] = adiabattemperatureflip[i-1]*np.exp(
            drflip[i-1]*alphaflip[i-1]*G*Mencflip[i-1]/(Cpflip*rflip[i-1]**2))
    adiabattemperature = np.flip(adiabattemperatureflip)
    temperature = adiabattemperature + \
        (rvar - rvar[-1])/(rvar[0]-rvar[-1])*deltaT
    # return the temperature at each layerboundary (including the surface)
    return temperature, adiabattemperature


# Layer step-wise calculating the adiabatic temperature profile
def topdownadiabat(rvar, drvar, surfacetemp, pressure, Menc):
    # flip the array of layerboundaries so it is ordered from the top down. r=rc excluded so first element cut before flip.
    rflip = np.flip(rvar[1:])
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc[1:])  # also flip the enclosed mass array
    # and the pressure array (innermost pressure is not needed)
    pressureflip = np.flip(centeraveraging(pressure))
    # calculate the flipped thermal expansion coefficient
    alphaflip = thermalexpansion(pressureflip)
    # create array for storing temperatures of the adiabatic temperature profile
    adiabattemperatureflip = np.zeros(number_of_structure_boundaries)
    # surface temperature for the first temperature of the adiabat
    adiabattemperatureflip[0] = surfacetemp
    for i in range(1, len(rvar)):
        # use previous layer's temperature to estimate the specific heat throughout the layer
        Cp = Cpfunc(adiabattemperatureflip[i-1])
        adiabattemperatureflip[i] = adiabattemperatureflip[i-1]*np.exp(
            drflip[i-1]*alphaflip[i-1]*G*Mencflip[i-1]/(Cp*rflip[i-1]**2))  # take a layer step in temperature
    return np.flip(adiabattemperatureflip)  # flip it back before returning


# Layer step-wise calculating the adiabatic temperature profile
def bottomupadiabat(rvar, drvar, coretemp, pressure, Menc):
    # calculate thermal expansion coefficient
    alpha = thermalexpansion(pressure)
    adiabattemperature = np.zeros_like(rvar)  # initialize return array
    # set bottom temp to be the core temperature
    adiabattemperature[0] = coretemp
    for i in range(1, len(rvar)):
        # use previous layer's temperature to estimate the specific heat throughout the layer
        Cp = Cpfunc(adiabattemperature[i-1])
        # take a layer step in temperature
        adiabattemperature[i] = adiabattemperature[i-1] * \
            np.exp(-drvar[i-1]*alpha[i-1]*G*Menc[i-1]/(Cp*rvar[i-1]**2))
    return adiabattemperature


# decide how (where) accretion heat is released. In each layer or at the CMB.
def accretionluminosity(accretedmassrate, rvar, drvar, Menc, Mode='slowfall'):
    if Mode == 'slowfall':  # accretion heat is released at each layer metal droplets pass through
        return accretionluminosityarray(accretedmassrate, rvar, drvar, Menc)
    else:  # accretion heat is released at the core mantle boundary
        return accretionluminositysingle(accretedmassrate, rvar, drvar, Menc)

# calculates the potential energy lost by accretion from infinity to the depth we consider the material deposits its heat
def accretionluminositysingle(accretedmassrate, rvar, drvar, Menc):
    # potential energy lost per unit time for accreting onto the surface
    surfaceterm = accretedmassrate*G*Menc[-1]/rsvar[-1]
    # potential energy lost per unit time from moving from the surface to the core mantle boundary
    CMBterm = accretedmassrate*G*np.sum(centeraveraging(Menc/rsvar**2)*drsvar)
    Lmet = surfaceterm+CMBterm  # total luminosity is the sum
    return Lmet

# luminosity from accretion if heat is transferred to the magma much quicker than the metal falling speed
def accretionluminosityarray(accretedmassrate, rvar, drvar, Menc):
    factors = accretedmassrate*G  # factors multiplying the integral
    Integral = np.cumsum(centeraveraging(Menc/rvar**2)*drvar)
    # accretion luminosity for all layer boundaries except surface boundary
    return factors*Integral

def accretionluminosityatmosphere(accretedmassrate, r, M): # luminosity from accreting full pebble mass to surface of planet
    return accretedmassrate*G*M/r

# calculates the Nusselt number based on a temperature gradient being able to transport the accretion luminosity and by assuming that the temperature gradient is the adiabatic temperature drop / magma ocean depth
def NufromL(Lmet, r, temperaturegradient, K=heatconductivity):
    Nu = Lmet/(4*pi*r**2*K*temperaturegradient)
    return Nu


# calculates the Nusselt number based on a temperature gradient being able to transport the accretion luminosity and by assuming that the temperature gradient is the adiabatic temperature drop / magma ocean depth
def RafromL(Lmet, r, temperaturegradient, K=heatconductivity):
    Nu = NufromL(Lmet, r, temperaturegradient, K)
    Racweak = 1000
    Rachard = 200
    # calculates Rayleigh number from inverse function of f: Ra --> Nu depending on weak or hard turbulence regime
    Ra = np.where(Nu > (1e19/Rachard)**2/7, Nu**(7/2)*Rachard, Nu**3*Racweak)
    return Ra


def NufromRa(Ra):
    Rachard = 200  # critical Rayleigh number for hard turbulence regime, AnatomyI
    Racweak = 1000
    # calculate the Nusselt Number depending on turbulence regime
    Nu = np.where(Ra > 1e19, (Ra/Rachard)**(2/7), (Ra/Racweak)**(1/3))
    return Nu+1  # +1 to account for heat conduction still occurring even if there were no convection


# calculate temperature difference between temperature profile and adiabatic temperature profile at the core using mean values for magma ocean variables
def deltaTadfromRa(Ra, r, pressure, temperature, density, Menc, K=heatconductivity, meanafter=True):
    L3 = (r[-1]-r[0])**3  # magma ocean depth cubed
    # calculate thermal expansion coefficient
    alpha = thermalexpansion(pressure)
    Cp = Cpfunc(temperature)  # calculate specific heat capacity
    g = G*Menc/r**2  # calculate local gravity
    if meanafter:  # mean of Rayleigh numbers
        deltaTad = np.mean(Ra*K/(alpha*Cp*density**2*g*L3))
    else:  # the Rayleigh number of means
        deltaTad = Ra*K/(np.mean(alpha)*np.mean(Cp) *
                         np.mean(density)**2*np.mean(g)*L3)
    return deltaTad


# adds a linear offset (deltaTad at core and 0 at surface) to the adiabatic temperature profile to estimate the real temperature profile
def temperaturefromdeltaTad(deltaTad, r, Tad):
    T = Tad + deltaTad*(r[-1]-r)/(r[-1]-r[0])
    return T


# integrates down from top of magma ocean to bottom to find the temperature profile
def equilibriumtemperature(Lmet, rvar, drvar, NuK, magmasurfacetemperature):
    rcenter = centeraveraging(rvar)  # find center of each layer
    if type(Lmet) == np.ndarray:  # check to see if integral has to be approximated
        Lmetflip = np.flip(Lmet)
        rcenterflip = np.flip(rcenter)
        drvarflip = np.flip(drvar)
        Integral = 1/(4*pi*NuK) * \
            np.flip(np.cumsum(Lmetflip/rcenterflip**2*drvarflip))
    else:  # or if there is an analytical solution
        # temperature integration from surface down to center of each layer
        Integral = Lmet/(4*pi*NuK)*(1/rcenter-1/rcenter[-1])
    Teq = np.append(magmasurfacetemperature+Integral, magmasurfacetemperature)
    return Teq


def NuKmax(Lmet, Tadcore, Tsurf, rcenter):  # calculates the upper limit for the effective heat conduction due to conduction + convection. (Can't result in smaller temperature drop than for the adiabatic temperature profile)
    if type(Lmet) == np.ndarray:
        drcenter = delta(rcenter)
        ret = 1/(4*pi*(Tadcore-Tsurf))*np.sum(centeraveraging(Lmet /
                                                              rcenter**2)*drcenter)  # approximate integral
    else:
        ret = Lmet*(1/rcenter[0]-1/rcenter[-1])/(4*pi*(Tadcore-Tsurf))
    return ret

# Legacy
# def equilibriumtemperature(pebblemetalluminosity, rvar, NuK, surfaceluminosity = 0): # solves linear system of equations for the layer temperatures at equilibrium
#     rcenter = centeraveraging(rvar)
#     delr = np.append(delta(rcenter), rvar[-1]-rcenter[-1])
#     matrixelements = 4*pi*NuK*rvar[1:]**2/delr
#     b = np.zeros_like(rcenter)
#     b[0] = -pebblemetalluminosity # core boundary condition set by accreted metals depositing their potential energy as heat at the core-mantle boundary
#     b[-1] = -surfaceluminosity # surface boundary condition
#     A = np.zeros((len(rcenter), len(rcenter))) # make matrix of zeros
#     A[0, :2] = np.array([-matrixelements[0], matrixelements[0]]) # add top row of matrix
#     A[-1, -2:] = np.array([matrixelements[-2], -matrixelements[-1]-matrixelements[-2]]) # add bottom row of matrix
#     for i in np.arange(1, len(b)-1):
#         A[i, i-1:i+2] = np.array([matrixelements[i-1], -matrixelements[i]-matrixelements[i-1], matrixelements[i]])
#     temperaturesol = np.linalg.solve(A, b)
#     checkbool = np.allclose(np.dot(A, temperaturesol), b)
#     return temperaturesol, checkbool

# def energymeltinglimit(Tmelt, layermass, H, temperaturestep): # calculates the energy limit needed for a layer to be liquid
#     energy = layermass*()
#     temperaturerange = np.append(np.arange(0, Tmelt, temperaturestep), Tmelt)

# def layertemperaturesol(layerenergy, layermass, temperaturestep): # calculates the temperature of a solid layer


# def layertemperatureliq(layerenergy, layermass, H, temperaturestep): # calculates the temperature of a liquid layer
#     LHS = layerenergy/layermass-H
#     if any (LHS<0):
#         raise Exception('Negative temperature found')
#     temperatureguess = 0
#     integral = 0
#     while integral < LHS:
#         previntegral = integral
#         integral+=Cpfunc(temperatureguess+0.5*temperaturestep)*temperaturestep
#         temperatureguess+=temperaturestep
#     temperatures = np.interp(LHS, [temperatureguess-temperaturestep, temperatureguess], [previntegral, integral])
#     return temperatures

# calculate the new density as a function of the bulk compression and thermal expansion
def densityfunc(pressure, temperature):
    pressureflip = np.flip(pressure)  # flip the pressure
    temperatureflip = np.flip(temperature)  # flip the temperature
    # calculate the thermal expansion coefficient assumed to be constant in each layer
    alphaflip = centeraveraging(thermalexpansion(pressureflip))
    # calculate the bulk modulus assumed to be constant in each layer
    Kflip = centeraveraging(bulkmodulus(pressureflip))
    densityflip = np.zeros(number_of_structure_boundaries)
    densityflip[0] = silicatedensity  # assume silicatedensity at top layer
    for i in range(1, number_of_structure_boundaries):
        densityflip[i] = densityflip[i-1]*np.exp((pressureflip[i]-pressureflip[i-1]) /
                                                 Kflip[i-1]-alphaflip[i-1]*(temperatureflip[i]-temperatureflip[i-1]))
    return np.flip(densityflip)


# calculate the new density by analytic integration of K w.r.t. pressure first (using linear K w.r.t pressure).
def densityfunc2(pressure, temperature):
    pressureflip = np.flip(pressure)  # flip the pressure
    temperatureflip = np.flip(temperature)  # flip the temperature
    # calculate the thermal expansion coefficient assumed to be constant in each layer
    alphaflip = centeraveraging(thermalexpansion(pressureflip))
    densityflip = np.zeros(number_of_structure_boundaries)
    densityflip[0] = silicatedensity  # assume silicatedensity at top layer
    for i in range(1, number_of_structure_boundaries):
        densityflip[i] = densityflip[i-1]*((K0+pressureflip[i]*K0prime)/(K0+pressureflip[i-1]*K0prime))**(
            1/K0prime)*np.exp(-alphaflip[i-1]*(temperatureflip[i]-temperatureflip[i-1]))
    return np.flip(densityflip)


# calculate the new boundaries for each layer given their new densities and old (conserved) masses
def newlayerboundaries(layermass, newdensity):
    rknew = np.zeros(number_of_structure_boundaries)
    rknew[0] = rc
    for i in range(1, number_of_structure_boundaries):
        rknew[i] = (3*layermass[i-1]/(4*pi*newdensity[i-1]) +
                    rknew[i-1]**3)**(1/3)
    return rknew


# interpolate the density from the new layer centers to the linearly spaced layer centers
def massanddensityinterpolation(r, dr, rnew, newdensity):
    # center of layers, where we want to know the density to calculate mass from
    rcenters = r[:-1]+dr/2
    # linearly interpolate w.r.t shell density such that total mass is preserved in interpolation
    density = np.interp(rcenters, rnew, rnew**2*newdensity)/rcenters**2
    layermass = density*4*pi*delta(r**3)/3
    return layermass, density


def Rayleigh(r, pressure, temperature, density, Menc, deltaTad, convectivedepth, K=heatconductivity):  # calculate Rayleigh number
    alpha = thermalexpansion(pressure)
    Cp = Cpfunc(temperature)
    g = G*Menc/r**2  # calculate local gravity
    # Dynamic viscosity. 100 Pa s in Pg, km, year units
    eta = 100*10**(-9)/GPaconversion*1/secondsperyear
    Ra = alpha*Cp*density**2*g*deltaTad * \
        convectivedepth**3/(K*eta)  # Rayleigh number
    return Ra


# calculate Rayleigh number
def RayleighMidpoint(r, pressure, temperature, density, Menc, deltaTad, convectivedepth, K=heatconductivity):
    length = len(r)
    if length % 2 == 0:  # even number of elements
        midind = [length//2-1, length//2]  # two midpoint indices
        # calculate values of the input variables at the middle of the magma ocean (as the mean of the two midpoint layers)
        rmidpoint = np.mean(r[midind])
        pressuremidpoint = np.mean(pressure[midind])
        temperaturemidpoint = np.mean(temperature[midind])
        densitymidpoint = np.mean(density[midind])
        Mencmidpoint = np.mean(Menc[midind])
    else:  # odd number of elements
        midind = length//2
        rmidpoint = r[midind]
        pressuremidpoint = pressure[midind]
        temperaturemidpoint = temperature[midind]
        densitymidpoint = density[midind]
        Mencmidpoint = Menc[midind]
    return Rayleigh(rmidpoint, pressuremidpoint, temperaturemidpoint, densitymidpoint, Mencmidpoint, deltaTad, convectivedepth, K)


def Nusselt(r, pressure, temperature, density, Menc, deltaTad, convectivedepth, midpoint=False):  # calculate the Nusselt number
    if midpoint:
        # calculate Rayleigh number at midpoint of magma ocean
        Ra = RayleighMidpoint(r, pressure, temperature,
                              density, Menc, deltaTad, convectivedepth)
    else:
        Ra = Rayleigh(r, pressure, temperature, density, Menc, deltaTad,
                      convectivedepth)  # calculate Rayleigh number locally
    Rachard = 200  # critical Rayleigh number for hard turbulence regime, AnatomyI
    Racweak = 1000
    # calculate the Nusselt Number depending on turbulence regime
    Nu = np.where(Ra > 1e19, (Ra/Rachard)**(2/7), (Ra/Racweak)**(1/3))
    return Nu+1  # + 1 to account for conduction still happening


def diffusivity(r, pressure, temperature, density, Menc, deltaTad, convectivedepth, K=heatconductivity, midpoint=False):  # calculate diffusivity
    Nu = Nusselt(r, pressure, temperature, density, Menc,
                 deltaTad, convectivedepth, midpoint)
    Cp = Cpfunc(temperature)
    return Nu*K/(density*Cp)  # km2 year-1


# calculate diffusivity when the variables are known on the layer boundaries
def diffusivitycentering(r, pressure, temperature, density, Menc, deltaTad, convectivedepth):
    rcenter = centeraveraging(r)
    centerpressure = centeraveraging(pressure)
    centertemperature = centeraveraging(temperature)
    cdensity = centerdensity(density, r)
    centerMenc = centeraveraging(Menc)
    return diffusivity(rcenter, centerpressure, centertemperature, cdensity, centerMenc, deltaTad, convectivedepth)

# initialization
starttime = time.time()

Msil0 = silicatedensity*4*pi*delta(rs**3)/3  # calculate initial layer mass
Msil = np.copy(Msil0)
Menc0 = menc(Msil0)  # calculate initial enclosed mass
# Does not update, same mass is always in each layer but the layer width differs from iteration to iteration
Menc = np.copy(Menc0)
gsurf = G*Menc[-1]/rs0**2  # gravity at surface

# atmosphere initialization
Ltot = accretionluminosityatmosphere(pebblerate, rs0, protomass) # full pebble luminosity 
monoatomicgamma = 5/3 # 1.666666
diatomicgamma = 7/5 # 1.4
triatomicgamma = 8/6 # 1.3333
lowgamma = 1.25 # higher temperatures --> lower Cp, lower Cp/Cv. 
atmgamma = lowgamma  # adiabatic index of the atmosphere
psurf = gsurf*Matm/(4*pi*rs0**2)*GPaconversion # GPa, pressure at surface
ppho = 2/3*gsurf/opacity*GPaconversion
Tpho = (Ltot/(4*pi*rs0**2*Boltzmann))**(1/4) # temperature at the base of the photosphere
Tsurf = Tpho*(psurf/ppho)**((atmgamma-1)/atmgamma)

# Tsurf = 3000 # K

# magma ocean initialization
# initial density is assumed to be constant
initialdensity = np.repeat(silicatedensity, number_of_structure_boundaries)
# initialpressure0 = psurf + GPaconversion*2*pi/3*silicatedensity**2*G*(rs**2-rs**2) # SOMETHING IS WRONG HERE. initial pressure in the magma ocean layerboundaries resulting from constant density profile, GPa
initialpressure = pressurefunc(rs, delta(rs), initialdensity, Menc)
# Alternate way to calc. pressure
initialpressure2 = pressurefunc2(rs, delta(rs), initialdensity, Menc)
initialtemperature, adiabattemperature = Tsurf+np.zeros(number_of_structure_boundaries), Tsurf+np.zeros(
    number_of_structure_boundaries)  # set cold initial temperature profile
pressurelong = np.copy(initialpressure)
temperaturelong = np.copy(initialtemperature)

for j in range(number_of_structure_timesteps):
    densitylong = densityfunc2(
        pressurelong, temperaturelong)  # calculate the density
    # resize the layers according to the new density
    rsvar = newlayerboundaries(Msil, densitylong)
    drsvar = delta(rsvar)
    pressurelong = pressurefunc(rsvar, drsvar, densitylong, Menc)
    Lmet = accretionluminosity(
        pebblerate*pebblemetalfrac, rsvar, drsvar, Menc, False)
    # calculate adiabatic temperature profile with the same surface temperature as the real temperature profile
    adiabattemperature = topdownadiabat(
        rsvar, drsvar, Tsurf, pressurelong, Menc)
    # calculate the adiabatic temperature gradient over the full magma ocean
    temperaturegradient = (
        adiabattemperature[0]-adiabattemperature[-1])/(rsvar[-1]-rc)
    # calculate the Rayleigh number
    Ra = RafromL(Lmet, rc, temperaturegradient, heatconductivity)
    print(Ra)
    deltaTad = deltaTadfromRa(
        Ra, rsvar, pressurelong, temperaturelong, densitylong, Menc, heatconductivity)
    temperaturelong = temperaturefromdeltaTad(
        deltaTad, rsvar, adiabattemperature)

    if (j % structureplotstepspacing == 0 or j == number_of_structure_timesteps-1) and compressionplotting:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        Mtotstr = str(np.round(Menc[-1]))
        fig.suptitle('Structure of magma ocean after '+str(j+1) +
                     ' timesteps. Total mass: '+Mtotstr+' Pg')

        colors = ['#1b7837', '#762a83', '#2166ac', '#b2182b']

        ax1.set_ylabel('Density [kg m-3]', color=colors[0])
        ax1.plot(rsvar, densityconversion*densitylong, color=colors[0])
        ax1.tick_params(axis='y', labelcolor=colors[0])

        ax3 = ax1.twinx()  # shared x-axis
        ax3.set_ylabel('Enclosed mass [M'+'$_{E}$]', color=colors[1])
        ax3.plot(rsvar, Menc/earthmass, color=colors[1])
        ax3.tick_params(axis='y', labelcolor=colors[1])

        color = 'tab:blue'
        ax2.set_xlabel('radius [km]')
        ax2.set_ylabel('Pressure [GPa]', color=colors[2])
        ax2.plot(rsvar, pressurelong, color=colors[2])
        ax2.tick_params(axis='y', labelcolor=colors[2])

        ax4 = ax2.twinx()  # shared x-axis
        color = 'tab:red'
        ax4.set_ylabel('Temperature [K]', color=colors[3])
        ax4.plot(rsvar, temperaturelong, color=colors[3])
        ax4.tick_params(axis='y', labelcolor=colors[3])
        # plot liquidus temperature alongside to see if the silicate is actually magma
        ax4.plot(rsvar, liquidus(pressurelong),
                 '--', color=colors[3], alpha=0.5)
        ax4.plot(rsvar, solidus(pressurelong), '--',
                 color=colors[3])  # plot solidus
        ax4.plot(rsvar, adiabattemperature, ':',
                 color=colors[3])  # plot adiabat

        ax1.locator_params(axis='both', nbins=6)  # set number of tick-marks
        ax2.locator_params(axis='both', nbins=6)
        ax1.grid(True)  # plot a grid
        ax2.grid(True)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

# calculate how much time in seconds is spent on calculating the initial differentiation of the planet
compressioncalctime = time.time()-starttime

# -----------------------------Parametrized functions---------------------------


# Carbon partition coefficient. Parametrization from Fischer2020. Pressure, temperature in GPa, K units.
def PCk(p, T, deltaIW=-2.2, Xsulfur=0, Xoxygen=0, NBOperT=2.6, microprobe=False):
    if microprobe: # derived from microprobe analysis in Fischer2020
            log10PCk = 1.81+2470/T-227*p/T+9.7 * \
            np.log10(1-Xsulfur)-30.6*np.log10(1-Xoxygen) - \
            0.123*NBOperT-0.211*deltaIW
    else: # derived from nanoSIMS analysis in Fischer2020
        log10PCk = 1.49+3000/T-235*p/T+9.6 * \
            np.log10(1-Xsulfur)-19.5*np.log10(1-Xoxygen) - \
            0.118*NBOperT-0.238*deltaIW
    return 10**log10PCk


# Nitrogen partition function. Parametrization from Grewal2019. Pressure, temperature in GPa, K units.
def PNk(p, T, XFeO,  Xsulfur=0, Xsilicon=0, NBOperT=2.6):
    a, b, c, d, e, f, g, h = -1415.40, 5816.24, 166.14, 343.44, -38.36, 139.52, 0.82, 1.13
    # calculate log10 of the partition coefficient for nitrogen.
    log10PNk = a+b/T+c*p/T+d * \
        np.log(1-Xsulfur)+e*np.log(1-Xsulfur)**2+f * \
        np.log(1-Xsilicon)+g*NBOperT+h*np.log(XFeO)
    return 10**log10PNk  # return the partition coefficient


# Water partition function. Parametrization from Luo2024. Function of water concentration in metal melt.
def PH2Ok(p, XH2O):
    log10PH2Ok = 1.26-0.34 * \
        np.log10(XH2O)+0.08*np.log10(XH2O)**2-4007/(1+np.exp((p+616)/80.6))
    return 10**log10PH2Ok


# -------------------------Restructuring and initializing-----------------------
# set the number of boundaries between layers of the magma ocean (also includes surface and core layer boundaries)
number_of_boundaries = 100

rs = rsvar[-1]  # new surface is the compressed surface
magmadepth = rs-rc  # km
rk, dr = np.linspace(rc, rs, number_of_boundaries, retstep=True)
# matrix delrk[k', k] with elements rk' - rk where rk' are the layer boundaries
delrk = np.array([rkprime-rk[:-1] for rkprime in rk[:-1]])
# matrix indexed [k', k] with elements rk' - rk where rk' are the layer midpoints
delrkcenter = np.array([rkprime-rk[:-1] for rkprime in centeraveraging(rk)])

number_of_layers = number_of_boundaries-1
# create array of distances from planetary center to center of each magma ocean sublayer
layercenters = rk[:-1]+dr/2
# calculate the volume of each layer
layervolumes = 4*pi*(rk[1:]**3-rk[:-1]**3)/3

# find the pressure, temperature and density at the new layer centers
pressure = np.interp(layercenters, rsvar, pressurelong)
temperature = np.interp(layercenters, rsvar, temperaturelong)

# initial volatile distribution
# initial mass fraction of volatiles, PLACEHOLDER
fi0 = 0.1*np.array([1/2, 1/3, 1/4, 1/5])
Pi0 = np.array([5, 300, 10, 100])  # H, C, N, O, PLACEHOLDER
# nominal values, PLACEHOLDER.
Pik0 = np.array([P0*np.ones_like(layercenters) for P0 in Pi0])

# masses
# Uncompressed silicate mass for plotting purposes
Msilunc = silicatedensity*layervolumes
Mencsilunc = menc(Msilunc)[:-1]  # uncompressed enclosed silicate mass

if constantdensity:
    sildensity = np.ones(number_of_layers)*silicatedensity
    density = sildensity + 
    Msil = silicatedensity*layervolumes
else:
    # Mass interpolated from compressed densities
    Msil, sildensity = massanddensityinterpolation(rk, dr, rsvar, densitylong)
    density = 
# constant mass [Pg] for the metal in any one layer resulting from isotropic accretion with constant falling speed.
Mmet = pebblerate*pebblemetalfrac*dr/vmet
massfrac = Mmet/Msil  # Metal silicate fraction in each magma ocean sublayer

if evendist:  # option 1: even concentration of volatiles
    # total mass of volatiles in each layer, Mik, [Pg]. PLACEHOLDER
    Mvol = np.array([[f0*M/(1-f0) for M in Msil+Mmet] for f0 in fi0])
elif innerdelta:  # option 2: delta function initial dist. of volatiles
    Mvol = np.zeros((len(volatiles), number_of_layers))
    Mvol[:, 0] = 1e10*fi0  # volatiles only exist in innermost layer

Mtot = np.copy(Msil)+Mmet+np.sum(Mvol, axis=0)
vmassinmet = Mvol*(1+1/(massfrac*Pik0))**(-1)

# Placeholder values for Diffusivity
# Dmin = 0.2*365.242199*86400/1e6 # km2 yr-1, PLACEHOLDER value stolen from AnatomyI for Vesta values
# Dmax = Dmin*1e3 # "2 to 3 orders of magnitude greater than for Vesta"

# calculate the diffusivity --> standard deviation
D = diffusivity(layercenters, pressure, temperature, density, centeraveraging(
    menc(Msil)), temperaturelong[0]-adiabattemperature[0], magmadepth)
Dmax = np.max(D)
stdmax = cspread*dr  # km
# years, set the timestep long enough that std spread is at most _ layers
dt = stdmax**2/(2*Dmax)
number_of_timesteps = int(simulatedtime/dt)  # compute the number of timesteps
plotstepspacing = max(plotstepspacing, int(number_of_timesteps/max_number_of_plots))

# %% Convection and sedimentation (transport)

# ------------------------------Transport functions-----------------------------


# function for computing new masses (of any element) in each magma ocean sublayer resulting from convection after a timestep dt
def massfromconvection(masses, sigmak, metsilfrac=0, Pik=0):
    # compute factor that depends only on previous layer (index k)
    konlyfactor = masses/2*(metsilfrac*Pik+1)**(-1)*sigmak*sqrt2/dr
    # compute the arguments for the function in the mass transfer equation
    x = delrk/(sigmak*sqrt2)  # matrix with index [k', k]
    xplus = (delrk+dr)/(sigmak*sqrt2)
    xminus = (delrk-dr)/(sigmak*sqrt2)
    if RBMdirk:
        drc = rk[:-1]-rc  # radial distance from core
        # set = 0 before adding sequence of terms for different n
        kprimefactor = np.zeros_like(x)
        for n in np.arange(-RBMnterms, RBMnterms+1):
            nrd = n*magmadepth  # compute n*(r_OB-r_IB)
            x += 2*nrd  # add n term to argument
            xplus += 2*nrd  # add n term to argument
            xminus += 2*nrd  # add n term to argument

            # create arguments for the extra terms added by considering RBM
            x2plusplus = (delrk+2*dr+2*nrd+2*drc)/(sigmak*sqrt2)
            x2plus = (delrk+dr+2*nrd+2*drc)/(sigmak*sqrt2)
            x2 = (delrk+2*nrd+2*drc)/(sigmak*sqrt2)
            kprimefactor += xplus*erf(xplus)-2*x*erf(x)+xminus*erf(xminus)+np.exp(-xplus**2)*sqrt2/sqrtpi-2*np.exp(-x**2)*sqrt2/sqrtpi+np.exp(-xminus**2)*sqrt2/sqrtpi\
                + x2plusplus*erf(x2plusplus)-2*x2plus*erf(x2plus)+x2*erf(x2)+np.exp(-x2plusplus**2)*sqrt2/sqrtpi-2*np.exp(-x2plus**2) * \
                sqrt2/sqrtpi + \
                np.exp(-x2**2)*sqrt2 / \
                sqrtpi  # Add the nth term in computing the factor depending on the new layer (index k')
    else:
        kprimefactor = xplus*erf(xplus)-2*x*erf(x)+xminus*erf(xminus)+np.exp(-xplus**2)*sqrt2/sqrtpi-2*np.exp(-x**2) * \
            sqrt2/sqrtpi+np.exp(-xminus**2)*sqrt2 / \
            sqrtpi  # compute the factor depending on the new layer (index k')

    # matrix ([k', k] format) of how mass from layer k has spread to layers k'
    Mkprimek = konlyfactor*kprimefactor
    if r2factor:
        Mkprimek *= 1/rk[:-1]**2  # rescale by rk^2
    if renormdist:
        Mkprimek *= np.divide(masses*(metsilfrac*Pik+1)**(-1), np.sum(Mkprimek, axis=0), out=np.zeros_like(masses),
                              where=np.sum(Mkprimek, axis=0) != 0)  # sum over all k' and upscale so that mass is conserved. FLAG HERE
    # compute mass in new layer k' as sum of mass contribution from each other layer k. (Axis = 0 is the k index axis).
    Mkprime = np.sum(Mkprimek, axis=1)
    if r2factor:
        Mkprime *= rk[:-1]**2  # rescale by rkprime^2
    if addtosource:
        # add back the mass that was lost. Compare the source mass to the spread of the source mass and re-add the lost mass.
        Mkprime += masses*(metsilfrac*Pik+1)**(-1)-np.sum(Mkprimek, axis=0)
    return Mkprime


# function for computing the mass of volatiles moved by convection (diffusion) with silicate melt over one time step
def massfromrelativeconvection(Mvol, Msil, Mmet, sigmak, Pvk=0):
    # mass of volatiles in silicate melt in a layer / silicate mass in that layer
    Xv0 = Mvol/(Msil+Pvk*Mmet)
    # create matrix [k', k] to be filled
    kprimefactor = np.zeros_like(delrkcenter)
    for n in np.arange(-RBMnterms, RBMnterms+1):
        # argument for positive erf functions, matrix with index [k', k]
        W = (delrkcenter+2*n*(rs-rc))/(sigmak*sqrt2)
        # argument for negative erf functions, matrix with index [k', k]
        Wminus = W-dr/(sigmak*sqrt2)
        # argument shift for second erf function
        Y = (2*(rk[:-1]-rc)+dr)/(sigmak*sqrt2)
        kprimefactor += erf(W)+erf(W+Y)-erf(Wminus)-erf(Wminus+Y)
    # density of volatiles in layer k' moved there by convection (diffusion) in silicate melt divided by the density of silicate in that layer
    Xvkprime = np.sum(Xv0/2*kprimefactor, axis=1)
    retmass = Xvkprime*Msil
    # find rescaling to conserve mass
    rescalingfactor = np.sum(retmass)/np.sum(Mvol)
    print(rescalingfactor)
    # approximate integral over one layer by calculating midpoint approximation of integral
    return Xvkprime*Msil/rescalingfactor

# The diffusion variable is r*Xv, not Xv.


# function for computing the mass of volatiles moved by convection (diffusion) with silicate melt over one time step
def massfromrelativeconvection2(Mvol, Msil, Mmet, sigmak, Pvk=0):
    # mass of volatiles in silicate melt in a layer / silicate mass in that layer
    Xv0 = Mvol/(Msil+Pvk*Mmet)
    # create matrix [k', k] to be filled
    kprimefactor = np.zeros_like(delrkcenter)
    for n in np.arange(-10*RBMnterms, 10*RBMnterms+1):
        # argument for positive erf functions, matrix with index [k', k]
        W = (delrkcenter+2*n*(rs-rc))/(sigmak*sqrt2)
        # argument for negative erf functions, matrix with index [k', k]
        Wminus = W-dr/(sigmak*sqrt2)
        # argument shift for second erf function
        Y = (2*(rk[:-1]-rc)+dr)/(sigmak*sqrt2)
        kprimefactor += erf(W)+erf(W+Y)-erf(Wminus)-erf(Wminus+Y)
    # density of volatiles in layer k' moved there by convection (diffusion) in silicate melt divided by the density of silicate in that layer
    Xvkprime = np.sum(layercenters*Xv0/2*kprimefactor, axis=1)/layercenters
    retmass = Xvkprime*Msil
    # incorrect BC means our solution yields mass gain/loss at inner/outer boundary
    # spurious mass gain entering at r = rc
    rcmassgain = (Xv0[0]+Xvkprime[0])*sigmak[0]**2 * \
        rk[0]*pi*Msil[0]/layervolumes[0]
    rsmassloss = (Xv0[-1]+Xvkprime[-1])*sigmak[-1]**2*rk[-1]*pi * \
        Msil[-1]/layervolumes[-1]  # spurious mass loss exiting at r = rs
    retmass[0] -= rcmassgain  # remove the mass gained
    retmass[-1] += rsmassloss  # add the mass lost
    # find rescaling to conserve mass
    rescalingfactor = np.sum(retmass)/np.sum(Mvol)
    print(rescalingfactor)
    # /rescalingfactor # approximate integral over one layer by calculating midpoint approximation of integral
    return Xvkprime*Msil


# explicit diffusion scheme for diffusive (convective) motion of volatiles
def explicitdiffusion(Mvol, Msil, Mmet, Drho, Pvk=0):
    # mass of volatiles in silicate melt in a layer / silicate mass in that layer
    Xv0 = Mvol/(Msil+Pvk*Mmet)
    # calculate difference between Xv for adjacent layers. Array is one element shorter than number of layers
    deltaXv = np.transpose(delta(np.transpose(Xv0)))
    # add CMB and surface boundary coundations (no flux)
    dXvdr = 1/dr*np.append(np.concatenate((np.array([0]), deltaXv)), 0)
    # net mass crossing boundary per dt (same length as number of boundaries)
    netinterface = 4*pi*rk**2*Drho*dXvdr
    deltaM = np.transpose(delta(np.transpose(netinterface)))*dt
    return Mvol+deltaM


# function for computing the movement of volatiles between layers as resulting from descending metal blobs.
def masschangefromsedimentation(masses, vmasses=Mvol, Pik=np.zeros(number_of_layers)):
    # calculate the number of layers that metal blobs descend through during one convective timestep dt.
    stepnumber = vmet*dt/dr
    # calculate descending volatile mass transfer from layers after the metal blobs descend 1 layerwidth.
    massdifference = vmasses[1:]*Pik[1:]*Mmet/(masses[1:]+(
        Pik[1:]-1)*Mmet) - vmasses[:-1]*Pik[:-1]*Mmet/(masses[:-1]+(Pik[:-1]-1)*Mmet)
    # calculate the volatile mass loss from the outermost layer
    outermassdifference = -vmasses[-1] * \
        Pik[-1]*Mmet/(masses[-1]+(Pik[-1]-1)*Mmet)
    outermassdifference = 0 # static from atmosphere interaction
    # return the change in volatile mass in each layer as the result of sedimentation descent
    return stepnumber*np.append(massdifference, outermassdifference)

# -------------------------Transport of volatiles loop--------------------------


# std = stdmax # Alt. 0: CONSTANT standard deviation of diffusion after a timestep dt for a point mass
# Alternative 1. Calculate diffusivity from Rayleigh number --> Nusselt number
std = np.sqrt(2*D*dt)
# std = stdmax-stdmax/2*np.sin(pi*(rk[:-1]-rc)/(rs-rc)) # Alt. 2. PLACEHOLDER sinusoidal dip from constant standard deviation
for j in range(number_of_timesteps):
    # copy how much volatile mass existed in previous timestep
    Mvolcopy = np.copy(Mvol)
    # Mvol = np.array([massfromconvection(Mvol[i], std, massfrac, Pik0[i])+vmassinmet[i]+masschangefromsedimentation(Mtot, Mvol[i], Pik0[i]) for i in irange]) # PLACEHOLDER, the added massfrac*vmass term corresponds to metal deposition speed = 0.
    sedimentation = np.array(
        [masschangefromsedimentation(Mtot, Mvol[i], Pik0[i]) for i in irange])
    if relativeconvection:
        Mvol = np.array([massfromrelativeconvection2(
            Mvol[i], Msil, Mmet, std, Pik0[i])+vmassinmet[i] for i in irange])+sedimentation
    elif explicit:
        # calculate product of diffusivity and density at layer boundaries. Assign a value of 0 at core mantle boundary and surface boundary (because it is being elementwise multiplied by diffusion flux across the boundary, which is 0 at the core mantle boundary and 0 at the surface either way)
        Drhocenter = np.append(np.concatenate(
            ([0], centeraveraging(D*density))), 0)
        Mvol = np.array([explicitdiffusion(Mvol[i], Msil, Mmet, Drhocenter,
                        Pik0[i])+vmassinmet[i] for i in irange])+sedimentation
    # calculate mass difference from previous volatile mass
    massdiff = np.sum(Mvolcopy)-np.sum(Mvol)

    vmassinmet = Mvol*(1+1/(massfrac*Pik0))**(-1)
    Mtot = Mmet+Msil+np.sum(Mvol, axis=0)  # total mass in each layer

    if j % plotstepspacing == 0 or j == number_of_timesteps-1:
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
        Mencsil = menc(Msil)[:-1]  # calculate enclosed silicate mass
        fig.suptitle('Mass and volatile distribution in magma ocean layers after '+"{:.1f}".format(
            (j+1)*dt)+' years, time step '+str(j+1)+'\nTotal volatile mass = '+str(np.round(Mvolsum))+' Pg', y=0.80)
        ax1.set(xlabel='Distance from planet center [km]', ylabel=(
            'Enclosed silicate mass [M'+'$_{E}$]'), aspect=3/4*(rs-rc)/(max(np.concatenate((Mencsil, Mencsilunc)))/earthmass))
        ax1.plot(layercenters, Mencsilunc/earthmass,
                 '-.', color='k', label='uncompressed')
        ax1.plot(layercenters, Mencsil/earthmass,
                 '.', color='k', label='compressed')
        ax1.set_ylim(bottom=0)
        ax1.legend()
        ax1.grid()

        ax2.set(xlabel='Distance from planet center [km]', ylabel=(
            'Volatile mass fraction'), aspect=3/4*(rs-rc)/np.max(np.matrix.transpose(Mvol/Msil)))
        ax2.set_prop_cycle('color', vcolors)
        ax2.plot(layercenters, np.matrix.transpose(
            Mvol/Msil), '.', label=volatiles)
        ax2.set_ylim(bottom=0)
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_ticks_position('right')
        ax2.grid()

        # # Plot densities
        # fig.suptitle('Density in magma ocean layers after '+"{:.1f}".format((j+1)*dt)+' years, time step '+str(j+1)+'\nTotal volatile mass = '+str(np.round(Mvolsum))+' Pg', y = 0.80)
        # ax1.set(xlabel='Distance from planet center [km]', ylabel=('Silicate density [Pg km-3]'), aspect=3/4*(rs-rc)/max(Msil/layervolumes))
        # ax1.plot(layercenters, M0sil/layervolumes, '-.', color='k', label='initial silicate density')
        # ax1.plot(layercenters, Msil/layervolumes, '.', color='k', label='silicate density')
        # ax1.set_ylim(bottom=0)
        # ax1.legend()
        # ax1.grid()

        # ax2.set(xlabel='Distance from planet center [km]', ylabel=('Volatile density [Pg km-3]'), aspect=3/4*(rs-rc)/np.max(Mvol/layervolumes))
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

# ---------------------------------Legacy---------------------------------------
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
