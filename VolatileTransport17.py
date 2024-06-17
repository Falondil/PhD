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
RBMdirk = True # decide to use analytical infinite series expression of diffusive motion from Dirk Veestraeten 2004. 
RBMnterms = 3 # decide how many positive and negative n terms of the infinite series expansion of 1/sinh to use in the PDF expression. 
renormdist = False # decide to use a truncated & renormalized normal distribution for the spread of mass
addtosource = False # decide if mass spread outside the inner and outer magma ocean boundaries should be re-added to the source layer
r2factor = True # decide to rescale the derived mass from convection transport equation by r^2 to turn 1D solution to 3D. 
cspread = 3 # allow the 1 std diffusion of particles from one layer to maximally spread _ layers
plotstepspacing = 3 # Every n timesteps plots are printed  

fixedplanetmass = True # incompatible with fixedplanetradius
fixedplanetradius = False # decide if the planet radius should be fixed between steps of iterative compression/expansion
compressionplotting = True # should the iterative steps to find the internal structure of the planet be plotted?
number_of_structure_timesteps = 10

if fixedplanetmass*fixedplanetradius:
    sys.exit('fixedplanetmass and fixedplanetradius cannot both be True.') 

# initializing
volatiles = np.array(['hydrogen', 'carbon', 'nitrogen', 'oxygen'])
vcolors = np.array(['c', 'r', 'g', 'b'])
irange = range(len(volatiles))

pi = np.pi
sqrtpi = np.sqrt(np.pi)
sqrt2 = np.sqrt(2)

# conversions
GPaconversion = 1.0041778e-15 # conversion to GPa from Pg, km, year units
densityconversion = 1e3 # converts from Pg km-3 to kg m-3
secondsperyear = 31556926 # 31 556 926 seconds per year

# units: years, km, Pg (Petagram)
simulatedtime = 1e3 # yr
rsurf = 6371 # km
coreradiusfraction = 3480/6371 # Earth's Core chapter by David Loper. PLACEHOLDER
rc = rsurf*coreradiusfraction # km
vmet = 1*secondsperyear/1e3 # km yr-1, falling speed of metal blobs
silicatedensity = 3.4 # Pg km-3 
coremetaldensity = 10.48 # 5.7 # Pg km-3, Earth's Core chapter by David Loper. PLACEHOLDER
mc = 4/3*pi*rc**3*coremetaldensity # Pg, mass of core
pebblemetalfrac = 0.3 # mass fraction of metal (iron) in accreted pebbles
# pebblerate = 4.7 # Pg yr-1. Calculated from model description from Marie-Luise Steinmayer 2023: Mdotaccretion = 2/3^(2/3)*(St=0.01/0.1)^(2/3)*sqrt((GMstar)=7.11e-5 au^3/yr^2)*(xi=0.003)*(Mgdot=1e-9 Msun/year)/(2*pi*((chi=15/14+3/14+3/2)+(3/2)*(alpha=1e-3))*(cs_0=0.137 au year-1)^2)*((Mplanet/Msun)=0.6*3e-6)^(2/3). Paste into google: # 2/3^(2/3)*(0.01/0.1)^(2/3)*sqrt(7.11e-5)*(0.003)*(1e-9)/(2*pi*((15/14+3/14+3/2)+(3/2)*(1e-3))*(0.137)^2)*(0.6*3e-6)^(2/3)
pebblerate = 1e-0*1e6 # Pg yr-1, PLACEHOLDER VALUE, x order(s) of magnitude less than what is required to form a 1 Earth mass planet over 5 Myr.
psurf = 1e11 # Pg km-1 yr-2. PLACEHOLDER VALUE. 1e11 = 1 atm.
G = 66465320.9 # gravitational constant, km3 Pg-1 yr-2

earthmass = 5.9712e12 # Earth's mass in Pg
# Alternative. Not bad but perhaps insufficient as a guarantee so better not unless it works
if fixedplanetmass:
    protoearthmass = 1*earthmass # Earth was less massive before Theia giant impact. Perhaps core mass should be lower too pre-giant impact.
    rsurf = ((protoearthmass-mc)/(4*pi*silicatedensity/3)+rc**3)**(1/3)
    
# compression and expansion values
alpha0 = 3e-5 # volumetric thermal expansion coefficient, K-1
K0 = 200 # bulk modulus, GPa
K0prime = 4 # bulk modulus derivative w.r.t pressure
psurf = 10/1e4 # surface pressure, GPa. corresponds to 10 bar. 
Tsurf = 2400 # surface temperature, K. 1600 K from Tackley2012, 3200 K from Monteux2016 PLACEHOLDER
deltaT = 1000 # temperature above that of an adiabat, K, PLACEHOLDER
Cpconv = 995839575 # converts specific heat capacity from units of J kg-1 K-1 to km2 yr-2 K-1
Cpsil = Cpconv*1200 # silicate melt heat capacity at constant pressure (1200 J kg-1 K-1), km2 yr-2 K-1 

number_of_structure_boundaries = int(1e4)
rs, drs = np.linspace(rc, rsurf, number_of_structure_boundaries, retstep=True) # value of the spherical coordinate r at each layer boundary

#------------------------------Structure functions-----------------------------
def delta(array): # calculates the forward difference to the next element in array. returns array with one less element than input array
    return array[1:]-array[:-1]    

def centeraveraging(array): # calculates the mean value between subsequent elements in array. returns array with one less element than input array
    return (array[1:]+array[:-1])/2

def centerdensity(densityatboundaries, r): # calculates the centerpoint density between density calculated at boundaries. 
    rcenters = r[:-1]+delta(r)/2 
    return np.interp(rcenters, r, r**2*densityatboundaries)/rcenters**2 # interpolate the shell density then divide by r^2 to get density. 

def menc(masses): # calculates mass enclosed below the lower boundary of each magma ocean layer
    return np.cumsum(np.concatenate(([0], masses)))+mc 

def thermalexpansion(pressure): # calculate the volumetric thermal expansion coefficient as a function of pressure, K-1
    return alpha0*(pressure*K0prime/K0+1)**((1-K0prime)/K0prime)

def bulkmodulus(pressure): # calculate the bulk modulus as a function of pressure
    return K0+pressure*K0prime

def Cpfunc(temperature, material='e'): # Shomate equation for specific heat capacity of material as function of temperature. NIST Webbook, Condensed phase thermochemistry data
    Tper1000 = temperature/1000 # Temperature/(1000 K) 
    if material=='e': # enstatite, MgSiO3
        molecularweight = 100.3887/1000 # kg/mol
        a1, b1, c1, d1, e1 = 146.4400, -1.499926e-7, 6.220145e-8, -8.733222e-9, -3.144171e-8 # 1850 K < T
        a2, b2, c2, d2, e2 = 37.72742, 110.1852, -50.79836, 7.968637, 15.16081 # 903 K < T < 1850 K
        Cpermole = np.where(temperature>1850, a1+b1*Tper1000+c1*Tper1000**2+d1*Tper1000**3+e1/Tper1000**2, a2+b2*Tper1000+c2*Tper1000**2+d2*Tper1000**3+e2/Tper1000**2) # Shomate equation, J/(mol*K))
    elif material=='f': # forsterite, Mg2Si04
        molecularweight = 140.6931/1000 # kg/mol
        a1, b1, c1, d1, e1 = 205.0164, -2.196956e-7, 6.630888e-8, -6.834356e-9, -1.298099e-7 # 2171 K < T
        a2, b2, c2, d2, e2 = 140.8749, 47.15201, -12.22770, 1.721771, -3.147210 # T < 2171 K
        Cpermole = np.where(temperature>2171, a1+b1*Tper1000+c1*Tper1000**2+d1*Tper1000**3+e1/Tper1000**2, a2+b2*Tper1000+c2*Tper1000**2+d2*Tper1000**3+e2/Tper1000**2) # Shomate equation, J/(mol*K))
    else:
        sys.exit('No silicate material matches input')
    return Cpermole/molecularweight*Cpconv # units of km2 year-2 K-1

def liquidus(pressure): # calculate the liquidus temperature as a function of pressure in high and low pressure regime
    def Tliquidus(T0, p0, q): #  calculate the liquidus temperature as a function of pressure
        return T0*(1+pressure/p0)**q # ref Monteux 2016 or AnatomyI. 
    return np.where(pressure<20, Tliquidus(1982.2, 6.594, 0.186), Tliquidus(2006.8, 34.65, 0.542)) # use low pressure function parameters if pressure is low and vice versa

def solidus(pressure): # calculate the liquidus temperature as a function of pressure in high and low pressure regime
    def Tsolidus(T0, p0, q): #  calculate the liquidus temperature as a function of pressure
        return T0*(1+pressure/p0)**q # ref Monteux 2016 or AnatomyI. 
    return np.where(pressure<20, Tsolidus(1661.2, 1.336, 1/7.437), Tsolidus(2081, 101.69, 1/1.226)) # use low pressure function parameters if pressure is low and vice versa

def pressurefunc(rvar, drvar, density, Menc): # calculates the pressure by integrating the density*gravitation down from the surface
    rflip = np.flip(rvar[:-1]) # flip the array of layerboundaries so it is ordered from the top down. r = rsurf excluded so last element cut before flip. 
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc[:-1]) # also flip the enclosed mass array
    densityflip = np.flip(centerdensity(density, rvar))
    return np.append(np.flip(psurf+GPaconversion*G*np.cumsum(densityflip*(2*pi/3*densityflip*(2*rflip*drflip+drflip**2)+(Mencflip-4*pi/3*rflip**3*densityflip)*(1/rflip-1/(rflip+drflip))))), psurf)

def pressurefunc2(rvar, drvar, density, Menc):
    rflip = np.flip(rvar[:-1]) # flip the array of layerboundaries so it is ordered from the top down. r = rsurf excluded so last element cut before flip. 
    drflip = np.flip(drvar)
    Mencflip = np.flip(Menc[:-1]) # also flip the enclosed mass array
    densityflip = np.flip(centerdensity(density, rvar))
    return np.append(np.flip(psurf+GPaconversion*G*np.cumsum(Mencflip*densityflip*drflip/rflip**2)), psurf)

def temperaturefunc(rvar, drvar, pressure, Menc): # calculate the temperature as a function of pressure and enclosed mass 
    rflip = np.flip(rvar[1:]) # flip the array of layerboundaries so it is ordered from the top down. r=rc excluded so first element cut before flip.
    drflip = np.flip(drvar) 
    Mencflip = np.flip(Menc[1:]) # also flip the enclosed mass array
    pressureflip = np.flip(centeraveraging(pressure)) # and the pressure array (innermost pressure is not needed)
    alphaflip = thermalexpansion(pressureflip) # calculate the flipped thermal expansion coefficient
    exponent = np.cumsum(G*alphaflip*Mencflip*drflip/(Cpsil*rflip**2)) # calculate the integral as a finite sum over each layer down to the layerboundary considered for the temperature
    return np.append(Tsurf*np.flip(np.exp(exponent)), Tsurf)+deltaT # return the temperature at each layerboundary (including the surface)

def temperaturefunc2(rvar, drvar, pressure, Menc): # Layer step-wise calculating the temperature as a function of pressure and enclosed mass. 
    rflip = np.flip(rvar[1:]) # flip the array of layerboundaries so it is ordered from the top down. r=rc excluded so first element cut before flip.
    drflip = np.flip(drvar) 
    Mencflip = np.flip(Menc[1:]) # also flip the enclosed mass array
    pressureflip = np.flip(centeraveraging(pressure)) # and the pressure array (innermost pressure is not needed)
    alphaflip = thermalexpansion(pressureflip) # calculate the flipped thermal expansion coefficient
    temperatureflip = np.zeros(number_of_structure_boundaries)
    temperatureflip[0] = Tsurf # surface temperature for the first temperature of the adiabat 
    for i in range(1, number_of_structure_boundaries):
        temperatureflip[i] = temperatureflip[i-1]*np.exp(drflip[i-1]*alphaflip[i-1]*G*Mencflip[i-1]/(Cpsil*rflip[i-1]**2))
    return np.flip(temperatureflip)+deltaT # return the temperature at each layerboundary (including the surface)

def temperaturefunc3(rvar, drvar, pressure, Menc): # Layer step-wise calculating the temperature as a function of pressure and enclosed mass using a variable specific heat capacity
    rflip = np.flip(rvar[1:]) # flip the array of layerboundaries so it is ordered from the top down. r=rc excluded so first element cut before flip.
    drflip = np.flip(drvar) 
    Mencflip = np.flip(Menc[1:]) # also flip the enclosed mass array
    pressureflip = np.flip(centeraveraging(pressure)) # and the pressure array (innermost pressure is not needed)
    alphaflip = thermalexpansion(pressureflip) # calculate the flipped thermal expansion coefficient
    temperatureflip = np.zeros(number_of_structure_boundaries)
    temperatureflip[0] = Tsurf # surface temperature for the first temperature of the adiabat 
    for i in range(1, number_of_structure_boundaries):
        Cpflip = Cpfunc(temperatureflip[i-1]+deltaT) # set the specific heat capacity to dependent on the temperature of the layer above
        temperatureflip[i] = temperatureflip[i-1]*np.exp(drflip[i-1]*alphaflip[i-1]*G*Mencflip[i-1]/(Cpflip*rflip[i-1]**2))
    return np.flip(temperatureflip)+deltaT # return the temperature at each layerboundary (including the surface)

def densityfunc(pressure, temperature): # calculate the new density as a function of the bulk compression and thermal expansion
    pressureflip = np.flip(pressure) # flip the pressure
    temperatureflip = np.flip(temperature) # flip the temperature
    alphaflip = centeraveraging(thermalexpansion(pressureflip)) # calculate the thermal expansion coefficient assumed to be constant in each layer
    Kflip = centeraveraging(bulkmodulus(pressureflip)) # calculate the bulk modulus assumed to be constant in each layer 
    densityflip = np.zeros(number_of_structure_boundaries)
    densityflip[0] = silicatedensity # assume silicatedensity at top layer
    for i in range(1, number_of_structure_boundaries):
        densityflip[i] = densityflip[i-1]*np.exp((pressureflip[i]-pressureflip[i-1])/Kflip[i-1]-alphaflip[i-1]*(temperatureflip[i]-temperatureflip[i-1]))
    return np.flip(densityflip)

def densityfunc2(pressure, temperature): # calculate the new density by analytic integration of K w.r.t. pressure first (using linear K w.r.t pressure).
    pressureflip = np.flip(pressure) # flip the pressure
    temperatureflip = np.flip(temperature) # flip the temperature
    alphaflip = centeraveraging(thermalexpansion(pressureflip)) # calculate the thermal expansion coefficient assumed to be constant in each layer
    densityflip = np.zeros(number_of_structure_boundaries)
    densityflip[0] = silicatedensity # assume silicatedensity at top layer
    for i in range(1, number_of_structure_boundaries):
        densityflip[i] = densityflip[i-1]*((K0+pressureflip[i]*K0prime)/(K0+pressureflip[i-1]*K0prime))**(1/K0prime)*np.exp(-alphaflip[i-1]*(temperatureflip[i]-temperatureflip[i-1]))
    return np.flip(densityflip)

def newlayerboundaries(layermass, newdensity): # calculate the new boundaries for each layer given their new densities and old (conserved) masses
    rknew = np.zeros(number_of_structure_boundaries)
    rknew[0] = rc
    for i in range(1, number_of_structure_boundaries):
        rknew[i] = (3*layermass[i-1]/(4*pi*newdensity[i-1])+rknew[i-1]**3)**(1/3)
    return rknew

def massanddensityinterpolation(r, dr, rnew, newdensity): # interpolate the density from the new layer centers to the linearly spaced layer centers
    rcenters = r[:-1]+dr/2 # center of layers, where we want to know the density to calculate mass from
    density = np.interp(rcenters, rnew, rnew**2*newdensity)/rcenters**2 # linearly interpolate w.r.t shell density such that total mass is preserved in interpolation
    layermass = density*4*pi*delta(r**3)
    return layermass, density

# initialization
starttime = time.time()

Msil0 = silicatedensity*4*pi*delta(rs**3)/3 # calculate initial layer mass 
Msil = np.copy(Msil0)
Menc0 = menc(Msil0) # calculate initial enclosed mass
Menc = np.copy(Menc0) 

initialdensity = np.repeat(silicatedensity, number_of_structure_boundaries) # initial density is assumed to be constant
initialpressure0 = psurf + GPaconversion*2*pi/3*silicatedensity**2*G*(rsurf**2-rs**2) # SOMETHING IS WRONG HERE. initial pressure in the magma ocean layerboundaries resulting from constant density profile, GPa
initialpressure = pressurefunc(rs, delta(rs), initialdensity, Menc)
initialpressure2 = pressurefunc2(rs, delta(rs), initialdensity, Menc) # Alternate way to calc. pressure
initialtemperature = temperaturefunc(rs, delta(rs), initialpressure, Menc)  # calculate the initial temperature in each layerboundary, unit K

pressure = np.copy(initialpressure)
temperature = np.copy(initialtemperature)

for j in range(number_of_structure_timesteps):
    density = densityfunc2(pressure, temperature) # calculate the density
    rsvar = newlayerboundaries(Msil, density) # resize the layers according to the new density
    if fixedplanetradius: # NOT FUNCTIONING AND PERHAPS LESS ELEGANT THAN FIXEDPLANETMASS
        pointstoadd = int((rsurf-max(rsvar))/drs)+1 # how many layer boundaries have to be added to reach the planet's surface
        rsvar = np.concatenate((rsvar, np.array([max(rsvar)+k*drs for k in range(1, pointstoadd)])))
        rsvar = np.append(rsvar, rsurf) # add the surface point
        Msil = np.concatenate((Msil, silicatedensity*4/3*pi*delta(rsvar[-pointstoadd-1:]**3))) # Really wrong for some reason. I don't know why. 
        Menc = menc(Msil)
        density = np.concatenate((density, np.repeat(silicatedensity, pointstoadd)))
    drsvar = delta(rsvar)
    pressure = pressurefunc(rsvar, drsvar, density, Menc)
    temperature = temperaturefunc3(rsvar, drsvar, pressure, Menc)
    
    if j%plotstepspacing==0 and compressionplotting:      
        fig, (ax1, ax2) = plt.subplots(1,2)
        Mtotstr = str(np.round(Menc[-1]))
        fig.suptitle('Structure of magma ocean after '+str(j+1)+' timesteps. Total mass: '+Mtotstr+' Pg')
        
        colors = ['#1b7837','#762a83','#2166ac','#b2182b']
        
        ax1.set_ylabel('Density [kg m-3]', color=colors[0])
        ax1.plot(rsvar, densityconversion*density, color=colors[0])
        ax1.tick_params(axis='y', labelcolor=colors[0])
        
        ax3 = ax1.twinx() # shared x-axis
        ax3.set_ylabel('Enclosed mass [M'+'$_{E}$]', color=colors[1])  
        ax3.plot(rsvar, Menc/earthmass, color=colors[1])
        ax3.tick_params(axis='y', labelcolor=colors[1])
        
        color = 'tab:blue'
        ax2.set_xlabel('radius [km]')
        ax2.set_ylabel('Pressure [GPa]', color=colors[2])
        ax2.plot(rsvar, pressure, color=colors[2])
        ax2.tick_params(axis='y', labelcolor=colors[2])
        
        ax4 = ax2.twinx() # shared x-axis
        color = 'tab:red'
        ax4.set_ylabel('Temperature [K]', color=colors[3])  
        ax4.plot(rsvar, temperature, color=colors[3])
        ax4.tick_params(axis='y', labelcolor=colors[3])
        ax4.plot(rsvar, liquidus(pressure), '--', color=colors[3], alpha=0.5) # plot liquidus temperature alongside to see if the silicate is actually magma
        ax4.plot(rsvar, solidus(pressure), '--', color=colors[3]) # plot solidus
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped    
        

compressioncalctime = time.time()-starttime # calculate how much time in seconds is spent on calculating the initial differentiation of the planet

# %% Convection and sedimentation (transport)

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

def PH2Ok(p, XH2O): # Water partition function. Parametrization from Luo2024. Function of water concentration in metal melt.
    log10PH2Ok = 1.26-0.34*np.log10(XH2O)+0.08*np.log10(XH2O)**2-4007/(1+np.exp((p+616)/80.6))
    return 10**log10PH2Ok

def diffusivity(r, pressure, temperature, density, Menc): # calculate diffusivity
    # use functions thermalexpansion(pressure), Cpfunc(temperature)
    centerdensity = centeraveraging(density)
    alpha = thermalexpansion(centeraveraging(pressure))
    Cp = Cpfunc(centeraveraging(temperature))
    g = centeraveraging(G*Menc/r**2) # calculate local gravity at center of layers
    K = 4*31425635.8 # base heat conductivity in AnatomyI, 4 W m-1 K-1 in Pg km year-3 K-1
    Rac = 200 # critical Rayleigh number, AnatomyI
    eta = 100*10**(-9)/GPaconversion*1/secondsperyear # Dynamic viscosity. 100 Pa s in Pg, km, year units
    Ra = alpha*Cp*centerdensity**2*g*deltaT*(max(r)-min(r))**3/(K*eta) # Rayleigh number
    Nu = (Ra/Rac)**(2/7) # Nusselt number
    return Nu*K/(centerdensity*Cp) # km2 year-1
    
#--------------------------Transport initialization----------------------------
number_of_boundaries = 100 # set the number of boundaries between layers of the magma ocean (also includes surface and core layer boundaries)

rsurf = rsvar[-1] # new surface is the compressed surface
magmadepth = rsurf-rc # km
rk, dr = np.linspace(rc, rsurf, number_of_boundaries, retstep=True)
delrk = np.array([rkprime-rk[:-1] for rkprime in rk[:-1]]) # matrix delrk[k', k] with elements rk' - rk. 

number_of_layers = number_of_boundaries-1
layercenters = rk[:-1]+dr/2 # create array of distances from planetary center to center of each magma ocean sublayer
layervolumes = 4*pi*(rk[1:]**3-rk[:-1]**3) # calculate the volume of each layer
fi0 = 0.1*np.array([1/2,1/3,1/4,1/5]) # initial mass fraction of volatiles, PLACEHOLDER
Pi0 = np.array([5, 300, 10, 100]) # H, C, N, O, PLACEHOLDER
Pik0 = np.array([P0*np.ones_like(layercenters) for P0 in Pi0]) # nominal values, PLACEHOLDER. 

# masses
Msilunc = silicatedensity*layervolumes # Uncompressed silicate mass for plotting purposes
Mencsilunc = menc(Msilunc)[:-1] # uncompressed enclosed silicate mass

Msil, _ = massanddensityinterpolation(rk, dr, rsvar, density) # Mass interpolated from compressed densities
Mmet = pebblerate*pebblemetalfrac*dr/vmet # constant mass [Pg] for the metal in any one layer resulting from isotropic accretion with constant falling speed. 
massfrac = Mmet/Msil # Metal silicate fraction in each magma ocean sublayer
# option 1: even concentration of volatiles
# Mvol = np.array([[f0*M/(1-f0) for M in Msil+Mmet] for f0 in fi0]) # total mass of volatiles in each layer, Mik, [Pg]. PLACEHOLDER
# option 2: delta function initial dist. of volatiles
Mvol = np.zeros((len(volatiles), number_of_layers))
Mvol[:, 0] = 1e10*fi0 # volatiles only exist in innermost layer

Mtot = np.copy(Msil)+Mmet+np.sum(Mvol, axis=0)
vmassinmet = Mvol*(1+1/(massfrac*Pik0))**(-1)

Dmin = 0.2*365.242199*86400/1e6 # km2 yr-1, PLACEHOLDER value stolen from AnatomyI for Vesta values
Dmax = Dmin*1e3 # "2 to 3 orders of magnitude greater than for Vesta"
stdmax = cspread*dr # km
dt = stdmax**2/(2*Dmax) # years, set the timestep long enough that std spread is at most _ layers
stdmin = np.sqrt(2*Dmin*dt) # km
number_of_timesteps = int(simulatedtime/dt) # compute the number of timesteps

#------------------------------Transport functions-----------------------------

def massfromconvection(masses, sigmak, metsilfrac=0, Pik=0): # function for computing new masses (of any element) in each magma ocean sublayer resulting from convection after a timestep dt 
    konlyfactor = masses/2*(metsilfrac*Pik+1)**(-1)*sigmak*sqrt2/dr # compute factor that depends only on previous layer (index k)
    # compute the arguments for the function in the mass transfer equation
    x = delrk/(sigmak*sqrt2) # matrix with index [k', k]
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
    if r2factor:
        Mkprimek *= 1/rk[:-1]**2 # rescale by rk^2
    if renormdist:
        Mkprimek *= np.divide(masses*(metsilfrac*Pik+1)**(-1), np.sum(Mkprimek, axis=0), out=np.zeros_like(masses), where=np.sum(Mkprimek, axis=0)!=0) # sum over all k' and upscale so that mass is conserved. FLAG HERE    
    Mkprime = np.sum(Mkprimek, axis=1) # compute mass in new layer k' as sum of mass contribution from each other layer k. (Axis = 0 is the k index axis).
    if r2factor:
        Mkprime *= rk[:-1]**2 # rescale by rkprime^2  
    if addtosource:
        Mkprime += masses*(metsilfrac*Pik+1)**(-1)-np.sum(Mkprimek, axis=0) # add back the mass that was lost. Compare the source mass to the spread of the source mass and re-add the lost mass.
    return Mkprime

def masschangefromsedimentation(masses, vmasses=Mvol, Pik=np.zeros(number_of_layers)): # function for computing the movement of volatiles between layers as resulting from descending metal blobs.
    stepnumber = vmet*dt/dr # calculate the number of layers that metal blobs descend through during one convective timestep dt. 
    massdifference = vmasses[1:]*Pik[1:]*Mmet/(masses[1:]+(Pik[1:]-1)*Mmet) - vmasses[:-1]*Pik[:-1]*Mmet/(masses[:-1]+(Pik[:-1]-1)*Mmet) # calculate descending volatile mass transfer from layers after the metal blobs descend 1 layerwidth.
    outermassdifference = -vmasses[-1]*Pik[-1]*Mmet/(masses[-1]+(Pik[-1]-1)*Mmet) # calculate the volatile mass loss from the outermost layer
    return stepnumber*np.append(massdifference, outermassdifference) # return the change in volatile mass in each layer as the result of sedimentation descent


#-------------------------Transport of volatiles loop--------------------------

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
        Mencsil = menc(Msil)[:-1] # calculate enclosed silicate mass
        fig.suptitle('Mass in magma ocean layers after '+"{:.1f}".format((j+1)*dt)+' years, time step '+str(j+1)+'\nTotal volatile mass = '+str(np.round(Mvolsum))+' Pg', y = 0.80)
        ax1.set(xlabel='Distance from planet center [km]', ylabel=('Enclosed silicate mass [M'+'$_{E}$]'), aspect=3/4*(rsurf-rc)/(max(np.concatenate((Mencsil,Mencsilunc)))/earthmass))
        ax1.plot(layercenters, Mencsilunc/earthmass, '-.', color='k', label='uncompressed')
        ax1.plot(layercenters, Mencsil/earthmass, '.', color='k', label='compressed')
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
        # fig.suptitle('Density in magma ocean layers after '+"{:.1f}".format((j+1)*dt)+' years, time step '+str(j+1)+'\nTotal volatile mass = '+str(np.round(Mvolsum))+' Pg', y = 0.80)
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