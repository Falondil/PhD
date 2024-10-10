# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:35:32 2024

@author: Vviik
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import time
from scipy.optimize import fsolve
from scipy.optimize import broyden1

# CHOICEs
fullformationtime = False # decide if the simulated time should be the time for accreting the mass of the planet
evendist = True  # decide initial distribution of volatiles to have a constant mass concentration throughout magma ocean
# decide initial distribution of volatiles to be concentrated in the layer closest to the core
innerdelta = not evendist

constantdensity = False  # decide to use the uncompressed silicate density throughout

number_of_boundaries = 100 # spatial discretization of volatile transport 
number_of_layers = number_of_boundaries-1
cspread = 0.1  # can't have too much spread if diffusion only transports to nearest layer
plotstepspacing = 1  # Every n timesteps plots are printed
# plot at most ___ plots. Redefines plotstepspacing if necessary
max_number_of_plots = 200

fixedplanetmass = True # incompatible with fixedplanetradius
structureplotting = True # should the iterative steps to find the internal structure of the planet be plotted?
number_of_structure_boundaries = int(1e4)
number_of_structure_iterations = 10
structureplotstepspacing = 1  # number_of_structure_timesteps

# -------------------------------General functions-----------------------------
def delta(array):  # calculates the forward difference to the next element in array. returns array with one less element than input array
    return array[1:]-array[:-1]


# calculates the mean value between subsequent elements in array. returns array with one less element than input array
def centeraveraging(array):
    return (array[1:]+array[:-1])/2

# initializing
volatiles = np.array(['hydrogen', 'carbon', 'nitrogen'])
vcolors = np.array(['c', 'r', 'g'])
irange = range(len(volatiles))
# volatiles = np.array(['hydrogen', 'carbon', 'nitrogen', 'oxygen'])
# vcolors = np.array(['c', 'r', 'g', 'b'])
# irange = range(len(volatiles))

pi = np.pi
sqrtpi = np.sqrt(np.pi)
sqrt2 = np.sqrt(2)

# conversions
GPaconversion = 1.0041778e-15  # conversion to GPa from Pg, km, year units
densityconversion = 1e3  # converts from Pg km-3 to kg m-3
secondsperyear = 31556926  # 31 556 926 seconds per year
Cpconv = 995839577 # converts specific heat capacity from units of J kg-1 K-1 to km2 yr-2 K-1
GPatobar = 1e4 # converts from GPa to bar
GPatokbar = GPatobar/1e3 # converts from GPa to kbar

# units: years, km, Pg (Petagram)
G = 66465320.9  # gravitational constant, km3 Pg-1 yr-2
Boltzmann = 1781.83355 # Boltzmann constant sigma = 5.67*1e-8 W/m^2 K^-4 converted to Petagram year^-3 K^-4

rcore = 3480 # km
coreradiusfraction = 3480/6371  # Earth's Core chapter by David Loper. PLACEHOLDER
vmet = 1*secondsperyear/1e3  # 1 m/s in km yr-1, falling speed of metal blobs
silicatedensity = 3.4 # Pg km-3
# 5.7 # Pg km-3, Earth's Core chapter by David Loper. PLACEHOLDER
coremetaldensity = 10.48
mc = 4/3*pi*rcore**3*coremetaldensity  # Pg, mass of core
earthmass = 5.9712e12  # Earth's mass in Pg
venusmass = 4.86732e12 # Venus' mass in Pg
# pebblerate = 4.7 # Pg yr-1. calculatesd from model description from Marie-Luise Steinmayer 2023: Mdotaccretion = 2/3^(2/3)*(St=0.01/0.1)^(2/3)*sqrt((GMstar)=7.11e-5 au^3/yr^2)*(xi=0.003)*(Mgdot=1e-9 Msun/year)/(2*pi*((chi=15/14+3/14+3/2)+(3/2)*(alpha=1e-3))*(cs_0=0.137 au year-1)^2)*((Mplanet/Msun)=0.6*3e-6)^(2/3). Paste into google: # 2/3^(2/3)*(0.01/0.1)^(2/3)*sqrt(7.11e-5)*(0.003)*(1e-9)/(2*pi*((15/14+3/14+3/2)+(3/2)*(1e-3))*(0.137)^2)*(0.6*3e-6)^(2/3)
pebblerate = 1*earthmass/5e6 # Pg yr-1, PLACEHOLDER VALUE, x order(s) of magnitude less than what is required to form a 1 Earth mass planet over 5 Myr.
pebblemetalfrac = 0.3  # mass fraction of metal (iron) in accreted pebbles

# Set planet mass, initial uncompressed radius is determined from CHOICE of planet mass
if fixedplanetmass:
    massscaling = 1
    # Earth was less massive before Theia giant impact. Perhaps core mass should be lower too pre-giant impact.
    protomass = massscaling*earthmass
    rsurf0 = ((protomass-mc)/(4*pi*silicatedensity/3)+rcore**3)**(1/3) # km, initial planet radius
else:
    rsurf0 = 6371  # km, initial planet radius
    
if fullformationtime: # rewrite simulatedtime to the time for the formation of the specified protoplanet's mass
    simulatedtime = protomass/pebblerate # year 
else:
    simulatedtime = 1e5 # yr
        
# compression and expansion values
alpha0 = 3e-5  # volumetric thermal expansion coefficient, K-1
# base heat conductivity in AnatomyI, 4 W m-1 K-1 in Pg km year-3 K-1
heatconductivity = 4*31425635.8
K0 = 200  # bulk modulus, GPa
K0prime = 4  # bulk modulus derivative w.r.t pressure
# silicate melt heat capacity at constant pressure (1200 J kg-1 K-1), km2 yr-2 K-1
Cpsil = Cpconv*1200
gsurf0 = G*protomass/rsurf0**2  # gravity at surface

# atmosphere
Mearthatm = 5.15e6  # Pg, current mass of Earth's atmosphere
Mvenusatm = 4.8e8 # Pg
Matmlow = Mearthatm*protomass/earthmass  # Pg, mass of the atmosphere of the protoplanet modelled
Matmhigh = Mvenusatm*protomass/venusmass
Matm = Matmhigh # CHOICE
opacitypho = 1e-3*1e6  # (gray) Rosseland mean opacity from Badescu2010. 1e6 factor for conversion from m^2 kg^-1 to km^2 Pg^-1, PLACEHOLDER
def opacityfunc(p, T): # scaling from Rogers2024
    alpha, beta = 0.68, 0.45
    coef = 1.29e-2*1e5*(1/1e-4)**alpha*(1/1000)**beta # 1e5 factor for conversion from cm^2 g^-1 to km^2 Pg^-1    
    kappa = coef*p**alpha*T**beta # kappa(p)
    return kappa

monoatomicgamma = 5/3 # 1.666666
diatomicgamma = 7/5 # 1.4
triatomicgamma = 8/6 # 1.3333
lowgamma = 1.25 # higher temperatures --> lower Cp, lower Cp/Cv. 
atmgamma = triatomicgamma

#------------------------------Structure functions-----------------------------
# calculates volume of all magma ocean layers using layer boundaries r
def volumefunc(r):
    return 4*pi*delta(r**3)/3


# calculates mass enclosed below each layer boundary
def menc(masses, mc = mc):  
    return np.cumsum(np.concatenate(([0], masses)))+mc


# calculates the centerpoint density between density calculatesd at boundaries.
def centerdensity(densityatboundaries, r):
    rcoreenters = r[:-1]+delta(r)/2
    # interpolate the shell density then divide by r^2 to get density.
    return np.interp(rcoreenters, r, r**2*densityatboundaries)/rcoreenters**2


# calculates the surface temperature assuming an adiabatic temperature profile from the photosphere down to the surface
def pTsurf(rsurf, pebblerate=pebblerate, protomass=protomass, Matm=Matmhigh, opacitypho=opacitypho, gamma=atmgamma):
    gsurf = G*protomass/rsurf**2
    Ltot = pebblerate*G*protomass/rsurf #+accretionluminositysingle(pebblerate*pebblemetalfrac, rs, drs, Menc0) # full pebble luminosity 
    psurf = gsurf*Matm/(4*pi*rsurf**2)*GPaconversion # GPa, pressure at surface
    ppho = 2/3*gsurf/opacitypho*GPaconversion
    Tpho = (Ltot/(4*pi*rsurf**2*Boltzmann))**(1/4) # temperature at the base of the photosphere
    Tsurf = Tpho*(psurf/ppho)**((gamma-1)/gamma)
    return psurf, Tsurf


# K-1, calculates the volumetric thermal expansion coefficient as a function in units of GPa of pressure. Abe1997
def thermalexpansion(pressure):
    return alpha0*(pressure*K0prime/K0+1)**((1-K0prime)/K0prime)


# Shomate equation for specific heat capacity of material as function of temperature. NIST Webbook, Condensed phase thermochemistry data
def Cpfunc(temperature, material='e'):  
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


# calculates the pressure by integrating the density*acceleration due to gravity down from the surface
def pressurefunc(rvar, drvar, density, Menc, psurf): 
    # flip the array of layerboundaries so it is ordered from the top down. r = rs excluded so last element cut before flip.
    rflip = np.flip(rvar[:-1])
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc[:-1])  # also flip the enclosed mass array
    densityflip = np.flip(centerdensity(density, rvar))
    return np.append(np.flip(psurf+GPaconversion*G*np.cumsum(densityflip*(2*pi/3*densityflip*(2*rflip*drflip+drflip**2)+(Mencflip-4*pi/3*rflip**3*densityflip)*(1/rflip-1/(rflip+drflip))))), psurf)


def topdownadiabat(rvar, drvar, pressure, Menc, Tsurf): # Layer step-wise calculating the adiabatic temperature profile
    # flip the array of layerboundaries so it is ordered from the top down. r=rcore excluded so first element cut before flip.
    rflip = np.flip(rvar[1:])
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc[1:])  # also flip the enclosed mass array
    # and the pressure array (innermost pressure is not needed)
    pressureflip = np.flip(centeraveraging(pressure))
    # calculates the flipped thermal expansion coefficient
    alphaflip = thermalexpansion(pressureflip)
    # create array for storing temperatures of the adiabatic temperature profile
    adiabattemperatureflip = np.zeros(number_of_structure_boundaries)
    # surface temperature for the first temperature of the adiabat
    adiabattemperatureflip[0] = Tsurf
    for i in range(1, len(rvar)):
        # use previous layer's temperature to estimate the specific heat throughout the layer
        Cp = Cpfunc(adiabattemperatureflip[i-1])
        adiabattemperatureflip[i] = adiabattemperatureflip[i-1]*np.exp(
            drflip[i-1]*alphaflip[i-1]*G*Mencflip[i-1]/(Cp*rflip[i-1]**2))  # take a layer step in temperature
    return np.flip(adiabattemperatureflip)  # flip it back before returning


# calculates the new density by analytic integration of K w.r.t. pressure first (using linear K w.r.t pressure).
def densityfunc(pressure, temperature, topdensity = silicatedensity):
    pressureflip = np.flip(pressure)  # flip the pressure
    temperatureflip = np.flip(temperature)  # flip the temperature
    # calculates the thermal expansion coefficient assumed to be constant in each layer
    alphaflip = centeraveraging(thermalexpansion(pressureflip))
    densityflip = np.zeros(number_of_structure_boundaries)
    densityflip[0] = topdensity
    for i in range(1, number_of_structure_boundaries):
        densityflip[i] = densityflip[i-1]*((K0+pressureflip[i]*K0prime)/(K0+pressureflip[i-1]*K0prime))**(
            1/K0prime)*np.exp(-alphaflip[i-1]*(temperatureflip[i]-temperatureflip[i-1]))
    return np.flip(densityflip)


# calculates new layerboundaries to keep mass inside each layer constant despite new densities
def newlayerboundaries(layermasses, newdensity):
    rnew = np.zeros(number_of_structure_boundaries)
    rnew[0] = rcore
    for i in range(1, number_of_structure_boundaries):
        rnew[i] = (3*layermasses[i-1]/(4*pi*newdensity[i-1]) +
                    rnew[i-1]**3)**(1/3)
    return rnew


# calculates the potential energy lost by accretion from infinity to the depth we consider the material deposits its heat
def accretionluminositysingle(accretedmassrate, rlimit, rvar, drvar, Menc):
    # potential energy lost per unit time for accreting onto the surface
    surfaceterm = accretedmassrate*G*Menc[-1]/rvar[-1]
    # potential energy lost per unit time from moving from the surface to the core mantle boundary
    
    MOterm = accretedmassrate*G*np.sum(np.where(centeraveraging(rvar)>rlimit, centeraveraging(Menc/rvar**2)*drvar, 0)) # integrates between rlimit and rsurf
    Lmet = surfaceterm+MOterm  # total luminosity is the sum
    print(str(surfaceterm), str(MOterm))
    return Lmet


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


# calculates temperature difference between temperature profile and adiabatic temperature profile at the core using mean values for magma ocean variables
def deltaTadfromRa(Ra, r, pressure, temperature, density, Menc, K=heatconductivity, meanafter=True):
    L3 = (r[-1]-r[0])**3  # magma ocean depth cubed
    # calculates thermal expansion coefficient
    alpha = thermalexpansion(pressure)
    Cp = Cpfunc(temperature)  # calculates specific heat capacity
    g = G*Menc/r**2  # calculates local gravity
    if meanafter:  # mean of Rayleigh numbers
        deltaTad = np.mean(Ra*K/(alpha*Cp*density**2*g*L3))
    else:  # the Rayleigh number of means
        deltaTad = Ra*K/(np.mean(alpha)*np.mean(Cp) *
                         np.mean(density)**2*np.mean(g)*L3)
    return deltaTad


# calculates the temperature profile by computing the adiabat temperature profile
# then calculating the temperature difference at the CMB and adding a linear slope
def temperaturefunc(rvar, drvar, rlimit, pressure, old_temperature, density, Menc, Tsurf, Lmet, K = heatconductivity, fulloutput = True):
    adiabattemperature = topdownadiabat(rvar, drvar, pressure, Menc, Tsurf) # calculates adiabatic temperature profile with the same surface temperature as the real temperature profile
    temperaturegradient = (adiabattemperature[0]-adiabattemperature[-1])/(rvar[-1]-rvar[0])
    Ra = RafromL(Lmet, rlimit, temperaturegradient, heatconductivity)
    print('Rayleigh number: '+str(Ra))
    deltaTad = deltaTadfromRa(Ra, rvar, pressure, old_temperature, density, Menc, heatconductivity)
    temperature = adiabattemperature + deltaTad*(rvar[-1]-rvar)/(rvar[-1]-rvar[0]) # add linear offset from adiabat temperature profile to match the core temperature offset
    if fulloutput:
        return temperature, adiabattemperature, Ra
    else:
        return temperature


# calculates the Nusselt number from the Rayleigh number in soft or hard turbulence regime
def NufromRa(Ra):
    Rachard = 200  # critical Rayleigh number for hard turbulence regime, AnatomyI
    Racweak = 1000
    # calculates the Nusselt Number depending on turbulence regime
    Nu = np.where(Ra > 1e19, (Ra/Rachard)**(2/7), (Ra/Racweak)**(1/3))
    return Nu+1  # +1 to account for heat conduction still occurring even if there were no convection


# calculates the diffusivity from the Nusselt number
def DfromNu(Nu, density, temperature, K=heatconductivity): 
    D = np.mean(Nu*K/(density*Cpfunc(temperature)))
    return D


# interpolate the density from the new layer centers to the linearly spaced layer centers, then calculate the new layermasses
def massanddensityinterpolation(r, rnew, newdensity):
    # center of layers, where we want to know the density to calculate mass from
    rcoreenters = centeraveraging(r)
    # linearly interpolate w.r.t shell density such that total mass is preserved in interpolation
    density = np.interp(rcoreenters, rnew, rnew**2*newdensity)/rcoreenters**2
    layermass = density*4*pi*delta(r**3)/3
    return layermass, density

    
# calculates p, T, rho profile and Ra when rho is known for the layers
def restructuringfromdensity(r, rho, pebblerate = pebblerate, number_of_structure_boundaries = number_of_structure_boundaries, iterations = number_of_structure_iterations, plotting = False):
    M = rho*volumefunc(r)  # calculates initial layer mass
    return restructuring(r, M, pebblerate, number_of_structure_boundaries, iterations, plotting)
    

# calculates p, T, rho profile and Ra when the layermasses are known 
def restructuring(r, M, pebblerate = pebblerate, number_of_structure_boundaries = number_of_structure_boundaries, iterations = number_of_structure_iterations, plotting=False):
    starttime = time.time()
    rvar = np.linspace(r[0], r[-1], number_of_structure_boundaries) # create refined boundaries between layers for the structure calculations
    drvar = delta(rvar)
    
    rho = M/volumefunc(r) # calculates the densities in the old layers
    initialdensity = np.interp(rvar, centeraveraging(r), rho) # interpolate to the new layers
    
    layermasses = centeraveraging(initialdensity)*volumefunc(rvar) # remains constant, only layer boundaries will shift with iterations
    Menc = menc(layermasses, mc) # enclosed mass. remains constant since mass inside each layer remains constant, only the layer thicknessses change with iteration
    
    psurf, Tsurf = pTsurf(rvar[-1], pebblerate, protomass) # calculate pressure and temperature at the surface of the world
    pressure = pressurefunc(rvar, drvar, initialdensity, Menc, psurf) # calculates initial pressure
    temperature = topdownadiabat(rvar, drvar, pressure, Menc, Tsurf) # calculates temperature
    
    for j in range(iterations):
        density = densityfunc(pressure, temperature, initialdensity[-1])
        rvar = newlayerboundaries(layermasses, density)
        drvar = delta(rvar)    

        rmiddle = 1/2*(rvar[-1]-rvar[0]) # center of magma ocean, use as the position where the accretion luminosity is supplied
        Lmet = accretionluminositysingle(pebblerate*pebblemetalfrac, rmiddle, rvar, drvar, Menc) # calculates the energy deposited into the magma ocean by the accretion of the metals in the pebbles 
        
        psurf, Tsurf = pTsurf(rvar[-1], pebblerate, protomass) # calculate pressure and temperature at the surface of the world
        pressure = pressurefunc(rvar, drvar, density, Menc, psurf)
        temperature, adiabattemperature, Ra = temperaturefunc(rvar, drvar, rmiddle, pressure, temperature, density, Menc, Tsurf, Lmet, heatconductivity, True)
        
        if (j % structureplotstepspacing == 0 or j == number_of_structure_iterations-1) and structureplotting:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            Mtotstr = str(np.round(Menc[-1]))
            fig.suptitle('Structure of magma ocean after '+str(j+1) +
                         ' timesteps. Total mass: '+Mtotstr+' Pg')

            colors = ['#1b7837', '#762a83', '#2166ac', '#b2182b']

            ax1.set_ylabel('Density [kg m-3]', color=colors[0])
            ax1.plot(rvar, densityconversion*density, color=colors[0])
            ax1.tick_params(axis='y', labelcolor=colors[0])

            ax3 = ax1.twinx()  # shared x-axis
            ax3.set_ylabel('Enclosed mass [M'+'$_{E}$]', color=colors[1])
            ax3.plot(rvar, Menc/earthmass, color=colors[1])
            ax3.tick_params(axis='y', labelcolor=colors[1])

            ax2.set_xlabel('radius [km]')
            ax2.set_ylabel('Pressure [GPa]', color=colors[2])
            ax2.plot(rvar, pressure, color=colors[2])
            ax2.tick_params(axis='y', labelcolor=colors[2])

            ax4 = ax2.twinx()  # shared x-axis
            ax4.set_ylabel('Temperature [K]', color=colors[3])
            ax4.plot(rvar, temperature, color=colors[3])
            ax4.tick_params(axis='y', labelcolor=colors[3])
            # plot liquidus temperature alongside to see if the silicate is actually magma
            ax4.plot(rvar, liquidus(pressure),
                     '--', color=colors[3], alpha=0.5)
            ax4.plot(rvar, solidus(pressure), '--',
                     color=colors[3])  # plot solidus
            ax4.plot(rvar, adiabattemperature, ':',
                     color=colors[3])  # plot adiabat

            ax1.locator_params(axis='both', nbins=6)  # set number of tick-marks
            ax2.locator_params(axis='both', nbins=6)
            ax1.grid(True)  # plot a grid
            ax2.grid(True)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped

    rshort = np.linspace(rvar[0], rvar[-1], len(r))
    rcentershort = centeraveraging(rshort)
    pressureshort = np.interp(rcentershort, rvar, pressure) 
    temperatureshort = np.interp(rcentershort, rvar, temperature)
    massshort, densityshort = massanddensityinterpolation(rshort, rvar, density)

    compressioncalctime = time.time()-starttime
    return rshort, psurf, Tsurf, pressureshort, temperatureshort, densityshort, Ra, compressioncalctime


#-----------------------------Initial Structure--------------------------------
# create magma ocean layer boundaries and calculate (p, T, rho) at the center of these layers
rk, dr0 = np.linspace(rcore, rsurf0, number_of_boundaries, retstep=True) # spatial discretization to be used for volatile transport
initialvolumes = volumefunc(rk) # calculate volume of each magma ocean layer 
initialMmet = pebblerate*pebblemetalfrac*dr0/vmet # metal inside each layer, LET PEBBLERATE CHANGE?
initialMtot = initialMmet+initialvolumes*silicatedensity
initialdensity = initialMtot/initialvolumes

# compressed structure
rk, psurf, Tsurf, pressure, temperature, density, Ra, compressioncalctime = restructuringfromdensity(rk, initialdensity, pebblerate, number_of_structure_boundaries, number_of_structure_iterations, True)
dr = rk[1]-rk[0] # same layerwidth for all layers
layercenters = centeraveraging(rk)
layervolumes = volumefunc(rk) # calculate volume of each magma ocean layer 
Mmet = pebblerate*pebblemetalfrac*dr/vmet # metal inside each layer, LET PEBBLERATE CHANGE?
Msilmet = density*layervolumes # compressed density*volume is total mass of metal + silicates
Msil = Msilmet-Mmet # subtract mass of metal to get mass of silicates in each magma ocean layer 
# uncompressed for plotting comparison 
Msilunc = silicatedensity*layervolumes
Mencsilunc = menc(Msilunc)[:-1]  # uncompressed enclosed silicate mass

rsurf = rk[-1]  # new surface is the compressed surface
magmadepth = rsurf-rcore # km

#-----------------------------Parametrized functions---------------------------
def IWfO2func(p, T): # oxygen fugacity of the IW-buffer Fe + O <-> FeO. Hirschmann2021. Valid for T>1000 K and P between 1e-4 and 100 GPa.
    def m(m0, m1, m2, m3, m4):
        return m0+m1*p+m2*p**2+m3*p**3+m4*p**(1/2)
    # parameters for fit to fcc (gamma-Fe) and bcc (alpha-Fe) allotropes of iron
    a = m(6.844864, 1.175691e-1, 1.143873e-3, 0, 0)
    b = m(5.791364e-4, -2.891434e-4, -2.737171e-7, 0, 0)
    c = m(-7.971469e-5, 3.198005e-5, 0, 1.059554e-10, 2.014461e-7)
    d = m(-2.769002e4, 5.285977e2, -2.919275, 0, 0)
    # parameters for fit to hcp (epsilon-Fe) allotrope of iron
    e = m(8.463095, -3.000307e-3, 7.213445e-5, 0, 0)
    f = m(1.148738e-3, -9.352312e-5, 5.161592e-7, 0, 0)
    g = m(-7.448624e-4, -6.329325e-6, 0, -1.407339e-10, 1.830014e-4)
    h = m(-2.782082e4, 5.285977e2, -8.473231e-1, 0, 0)

    hcpdomain = p>(-18.64 + 0.04359*T - 5.069e-6*T**2) # check for what p, T the iron is hcp
    def fit(P1, P2, P3, P4): # fitting function using either (a,b,c,d) or (e,f,g,h)
        return P1+P2*T+P3*T*np.log(T)+P4/T
    log10fO2 = np.where(hcpdomain, fit(e,f,g,h), fit(a,b,c,d))
    return 10**log10fO2


# Carbon partition coefficient. Parametrization from Fischer2020. Pressure, temperature in GPa, K units.
def PCfunc(p, T, deltaIW=-2.2, Xoxygen=0, Xsulfur=0, NBOperT=2.6, microprobe=False):
    if microprobe: # derived from microprobe analysis in Fischer2020
            log10PC = 1.81+2470/T-227*p/T+9.7 * \
            np.log10(1-Xsulfur)-30.6*np.log10(1-Xoxygen) - \
            0.123*NBOperT-0.211*deltaIW
    else: # derived from nanoSIMS analysis in Fischer2020
        log10PC = 1.49+3000/T-235*p/T+9.6 * \
            np.log10(1-Xsulfur)-19.5*np.log10(1-Xoxygen) - \
            0.118*NBOperT-0.238*deltaIW
    return 10**log10PC


# Nitrogen partition function. Parametrization from Grewal2019. Pressure, temperature in GPa, K units.
def PNfunc(p, T, XFeO,  Xsulfur=0, Xsilicon=0, NBOperT=2.6):
    a, b, c, d, e, f, g, h = -1415.40, 5816.24, 166.14, 343.44, -38.36, 139.52, 0.82, 1.13
    # calculate log10 of the partition coefficient for nitrogen.
    lnPN = a+b/T+c*p/T+d*np.log(100*(1-Xsulfur))+e*np.log(100*(1-Xsulfur))**2+f*np.log(100*(1-Xsilicon))+g*NBOperT+h*np.log(100*XFeO)
    return np.exp(lnPN)  # return the partition coefficient


# Water partition function. Parametrization from Luo2024. Function of water concentration in metal melt.
def PH2Ofunc(p, XH2Omet):
    log10PH2O = 1.26-0.34 * \
        np.log10(XH2Omet)+0.08*np.log10(XH2Omet)**2-4007/(1+np.exp((p+616)/80.6))
    return 10**log10PH2O


# mass of water in metal droplets/mass of metal droplets, (torn long-edge @ page 8 of block C for derivation) 
def watermetalfraction(PH2O, MH2O, Msil, Mmet=Mmet):
    return PH2O*MH2O/(PH2O*Mmet+Msil) 


# calculates PH2O then watermetalfraction, then back to PH2O using a starting guess for the mass fraction of water in the metal droplets.
def iterativePH2O(p, MH2O, Msil, Mmet=Mmet, CH2Ometguess=0.01, iterations=10):
    for j in range(iterations):
        PH2Oguess = PH2Ofunc(p, CH2Ometguess)
        CH2Ometguess = watermetalfraction(PH2Oguess, MH2O, Msil, Mmet)
    return PH2Oguess, CH2Ometguess


# calculate oxygen fugacity of IW buffer
IWfO2 = IWfO2func(pressure, temperature) 

# plot oxygen fugacity as function of p, T. Add our own p, T curve. 
X, Y = np.meshgrid(pressure, temperature)
Z = np.array([np.log10(IWfO2func(p,t)) for p in pressure for t in temperature]).reshape(number_of_layers, number_of_layers)
IWfig = plt.figure()
CS = plt.contourf(X,Y,Z,20)
cbar = IWfig.colorbar(CS)
plt.plot(pressure, temperature, color='k') #, norm=pltcolors.TwoSlopeNorm(0), cmap='bwr' can be used
# invert axes so that going to the left or going down on the plot corresponds to further into the magma ocean
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.title('log Oxygen fugacity at the IW-buffer')
plt.xlabel('Pressure [GPa]')
plt.ylabel('Temperature [K]')

#------------------------------Volatile Transport------------------------------
def massfraction(partialmasses, layermasses): # computes mass fraction of a mass compared to the total
    return partialmasses/(layermasses+np.sum(partialmasses, 0))


#----------------atmosphere & magma ocean top layer interaction----------------

def molefraction(wi, amui, wj, amuj): # takes weight fraction of volatile species i and calculates the mole fraction of species i using the atomic mass of i as well as the weight fraction and atomic mass of all other volatile species j (wj and amuj are arrays)
    return (wi/amui)/(wi/amui+np.sum(wj*amuj))

# molar masses of volatile gas species
amuH2O = 18 
amuCO2 = 44 
amuN2 = 28 
amui = np.array([amuH2O, amuCO2, amuN2])

# molar masses of the atoms
amuC = 12
amuN = 14

# molar masses of melt species
amuOH = 17
amuCO3 = 60
amuNH = 15
amuNH2 = 16
amuNH3 = 17

# melt properties
melt = 'enstatite'
NBOperT = 2.2 # CHOICE
if melt=='enstatite': # MgSiO3 simplified. Same resulting product if considering Mg2Si2O6 instead since amuMelt is doubled and Siperformula becomes is halved. 
    amuMelt = 100 
    TperSi = 1 # one Mg for each Si
    Siperformula = 1
else: # forsterite, Mg2SiO4
    amuMelt = 140
    TperSi = 2 # two Mg for each Si
    Siperformula = 1 # 1 Si in the unit formula

# number of volatile-bearing products per volatile reactant
alphaH2O = 2 # H2O(gas) + O(melt) = 2 OH(melt) e.g. Stolper1982. 
alphaCO2 = 1 # CO2(gas) + O(melt) = CO3(melt) e.g. Ortenzi2016, Fegley2020. Or more accurately M2SiO4(melt)+CO2(g) = MCO3(melt) + MSiO3(melt) e.g. Spera1980. 
alphaNH_ = 2 # N2(gas)+_*H2O(gas)+(3-_)*O(melt) = 2 NH_ + 3/2 O(gas) e.g. Grewal2020.
alphai = np.array([alphaH2O, alphaCO2, alphaNH_]) 

# conversion factor from fraction of NBO that get a volatile attached to them to mass fraction of the volatile atoms inside the melt
betai = NBOperT*TperSi*Siperformula/amuMelt*np.array([amuH2O, amuC, amuN]) # PLACEHOLDER, using amuH2O to count H. 

# equilibrium coefficients
KH2O = 0.1*np.exp(-3+(psurf*GPatokbar)/10) # Stolper1982, fig. 4 and eq. 8. Product of K1 and K2. Slopes up one log unit after 10 kbar.
KCO2withNepheline = np.exp(-12.5-5.5*psurf*GPatokbar/30) # Spera1980, fig. 1. Minus sign since we want the reverse equilibrium constant
KCO2withOlivineMelilite =  np.exp(-13-5*psurf*GPatokbar/30) # Spera1980, fig. 3. -||-
KCO2 = KCO2withNepheline # CHOICE
KN2 = KH2O*1e-8 # PLACEHOLDER. FIND A REAL VALUE

Ki = np.array([KH2O, KCO2, KN2])

# initial mass fraction of volatiles in melt, PLACEHOLDER
XH2O = 2e3*1e-6 # 2e3 ppmw, AnatomyIII
XC = 600*1e-6 # 600 ppmw total estimated from Anders2021 or AnatomyIII. 27 ppmw of total planet mass of Venus lies in atmosphere (google) 
XN = 10**0.5*1e-6 # mass fraction of total planet mass of Venus in atmosphere, from same figure in AnatomyIII
XH = XH2O*2/18 # PLACEHOLDER DOES THIS EVEN MAKE SENSE, 2 of the 18 amu in H2O is from the hydrogen
XFeO = 0.3*(55.845+16)/55.845 # PLACEHOLDER DOES THIS EVEN MAKE SENSE. 30% of bulk mass is metal. 
Xi = np.array([XH2O, XC, XN]) # array to store the volatiles in.

# initial masses each volatile element in the silicate melt
MH2O = XH2O*Msil
MC = XC*Msil
MN = XN*Msil
Mvol = np.array([MH2O, MC, MN])
Mtot = Msil+MH2O+MC+MN

# Matm - mass of the atmosphere
# Miatm0 - starting masses of each volatile species in the atmosphere
# Misil0 - starting masses of each volatile species in the top layer of the magma ocean (silicate melt)
# Mitot0 - starting masses of each volatile species in either the atmosphere of top layer of the magma ocean
# AMUi - atomic mass units of volatile species i
# alphai - vector for the powers for the dissolution law of the volatile species i's dissolution into the magma ocean
# Ki - equilibrium constants for the dissolution reactions
# Msilmet - silicate + metal mass in top layer of magma ocean 
# Xiktop - mass fraction of volatiles in the top layer of the magma ocean

# returns masses of each volatile in the first atmosphere to be in equilibrium with the assumed starting volatile fraction in the silicate melt of the top layer of the magma ocean
def startingatmosphere(Matm, Xiktop, AMUi, alphai, Ki):
    frac = Xiktop**alphai*AMUi/Ki
    Miatm0 = frac/np.sum(frac)*Matm
    return Miatm0

# Possibly useful. calculates the Ki corresponding to if the non-equilbriated dissolution was in fact an equilibrium
def Kinoneq(Xiktop, Miatm0, AMUi, alphai): 
    frac = Miatm0/AMUi
    return Xiktop**alphai/(frac/np.sum(frac))

# # input needs to be arrays, sensitive to actual values of input parameters (must be fairly close to equilibrium already) be careful
# def legacyMiatmeqfunc(Miatm0, AMUi, alphai, Ki, Msilmet, Misil0, method='fsolve'): # function to find the dissolution equilbrium solution of the mass of volatile species in the atmosphere 
#     Mitot0 = Miatm0+Misil0 # sum of volatiles in either reservoir before equilibriation
#     def Miatmrootfunc(Miatm): # function of Miatm that evaluates to 0 at chemical dissolution equilbrium between top layer of magma ocean and the atmosphere
#         Mtot0 = Msilmet+np.sum(Mitot0) # total mass of the top layer before equilibriation
#         frac = Miatm/AMUi
#         # return Miatm+(Ki*frac/np.sum(frac))**(1/alphai)*(Mtot0-np.sum(Miatm))-Mitot0
#         return 1/Mitot0*(Miatm+(Ki*frac/np.sum(frac))**(1/alphai)*(Mtot0-np.sum(Miatm)))-1 # better to solve for the unitless equation = 0
#     if method=='fsolve': # appears to be stably solver further from equilibrium starting guesses
#         Miatmeq = fsolve(Miatmrootfunc, Miatm0) # find roots to the function with a starting guess being the unequilibrated atmospheric mass of the volatile species.
#     else:
#         Miatmeq = broyden1(Miatmrootfunc, Miatm0, f_tol=1e-8) # use Broyden's good method instead
#     print(np.isclose(Miatmrootfunc(Miatmeq), np.zeros_like(Miatmeq))) # check if roots are valid
    
#     Misileq = Mitot0-Miatmeq # mass of volatile in magma ocean = total volatile mass minus volatile mass in atmosphere
#     Xiktop = massfraction(Misileq, Msilmet) # mass fraction of volatile in top layer of magma ocean
#     return Miatmeq, Misileq, Xiktop, Miatmeq/Miatm0


# input needs to be arrays, sensitive to actual values of input parameters (must be fairly close to equilibrium already) be careful
def Miatmeqfunc(Miatm0, AMUi, alphai, betai, Ki, Msiltop, Misil0, method='fsolve'): # function to find the dissolution equilbrium solution of the mass of volatile species in the atmosphere 
    Mitot0 = Miatm0+Misil0 # sum of volatiles in either reservoir before equilibriation
    def Miatmrootfunc(Miatm): # function of Miatm that evaluates to 0 at chemical dissolution equilbrium between top layer of magma ocean and the atmosphere
        frac = Miatm/AMUi
        return 1/Mitot0*(Miatm+(Ki*frac/np.sum(frac))**(1/alphai)*betai*Msiltop)-1 # better to solve for the unitless equation = 0
    if method=='fsolve': # appears to be stably solver further from equilibrium starting guesses
        Miatmeq = fsolve(Miatmrootfunc, Miatm0) # find roots to the function with a starting guess being the unequilibrated atmospheric mass of the volatile species.
    else:
        Miatmeq = broyden1(Miatmrootfunc, Miatm0, f_tol=1e-8) # use Broyden's good method instead
    print(np.isclose(Miatmrootfunc(Miatmeq), np.zeros_like(Miatmeq))) # check if roots are valid
    
    Misileq = Mitot0-Miatmeq # mass of volatile in magma ocean = total volatile mass minus volatile mass in atmosphere
    Xiktop = massfraction(Misileq, Msiltop) # mass fraction of volatile in top layer of magma ocean
    return Miatmeq, Misileq, Xiktop, Miatmeq/Miatm0


Miatmstart = startingatmosphere(Matm, (Mvol[:,-1]/Mtot[-1])/betai, amui, alphai, Ki)


#-------------------convection (diffusion) and sedimentation-------------------

# estimate initial partition coefficients using bulk mass fractions globally
if evendist: 
    PH2O, CH2Omet = iterativePH2O(pressure, MH2O, Msil)
    PC = PCfunc(pressure, temperature, 0)
    PN = PNfunc(pressure, temperature, XFeO)
    Pvol = np.array([PH2O, PC, PN]) # array of all the volatile species' partition coefficients (at all magma ocean layers)
    
    plt.figure()
    plt.gca().set_prop_cycle('color', vcolors)
    plt.semilogy(layercenters, np.transpose(Pvol), label=volatiles)
    plt.xlabel('radius [km]')
    plt.ylabel('Partition coefficient')
    plt.title('Metal droplets/silicate melt partition coefficient of volatile species')
    plt.legend()

    plt.figure()
    plt.gca().set_prop_cycle('color', vcolors)
    plt.semilogy(pressure, np.transpose(Pvol), label=volatiles)
    plt.xlabel('pressure [GPa]')
    plt.ylabel('Partition coefficient')
    plt.title('Metal droplets/silicate melt partition coefficient of volatile species')
    plt.legend()

    # plt.figure()
    # plt.plot(layercenters, CH2Omet, 'k')
    # plt.gca().set_ylim(bottom=0)
    # plt.ylabel('Water fraction inside metal droplets')
    # plt.xlabel('radius [km]')


