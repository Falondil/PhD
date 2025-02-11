"""
Created on Mon Sep 16 11:35:32 2024

@author: Vviik
"""
# %% python -m cProfile myscript.py # Run this in the terminal to get timings for different function calls

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mplcolors 
import time
from scipy.optimize import fsolve
from scipy.optimize import broyden1
from scipy.linalg import eig
from scipy.linalg import inv

# CHOICES
nosedifrac = 0.1 # for how big fraction of the total time should sedimentation be halted. CHOICE 0 to 1. 
timechoice = False # should settime be the CHOICE
masschoice = True # or should Mfinal be the CHOICE
bothchoice = timechoice == masschoice # If timechoice == masschoice then both are CHOICES. (Both being False also works here.)
constantpebblerate = False # decide if the planet mass should grow linearly or exponentially
evendist = True  # decide initial distribution of volatiles to have a constant mass concentration throughout magma ocean
# decide initial distribution of volatiles to be concentrated in the layer closest to the core
innerdelta = not evendist
growthbool = True # decide if the planet should grow (met. and sil. accretion)
constantdensity = False  # decide to use the uncompressed silicate density throughout

number_of_boundaries = 70 # spatial discretization of volatile transport CHOICE
number_of_layers = number_of_boundaries-1
# cspread = 0.1  # Alt. 0. Allow 10 sigma inaccuracy. can't have too much spread if diffusion only transports to nearest layer
cspread = 0.25 # Alt. 1. Allow 4 sigma inaccuracy.
plotstepspacing = 1  # Every n timesteps plots are printed
# plot at most ___ plots. Redefines plotstepspacing if necessary
max_number_of_plots = 300 # CHOICE, 400 seems to be just too big for the cluster job memory limit? slurmstepd: error: StepId=49244262.batch exceeded memory limit (6679385088 > 6287261696), being killed
max_len_savefiles = 10*max_number_of_plots # CHOICE

structureplotting = True # should the iterative steps to find the internal structure of the planet be plotted?
number_of_structure_boundaries = int(1e4)
number_of_structure_iterations = 10
structureplotstepspacing = 1  # number_of_structure_timesteps
restructuringspacing = 10

M0 = 1e-4 # mass at which mass first starts accreting pebbles (units = Earth masses). 1e-2 I read somewhere(?), 1e-4 from Anders or maybe 1e-6 from Claudia. 
formationtime = 5e6 # time [yr] needed to form one Earth mass planet

massscaling = 0.2 # how many earth masses to start with. >0.3 seems very safe (deltaTad = negligible), >0.1 might be safe (temperature profile looks plausible).
stoppingmass = 0.6 # after how many earth masses do we stop accretion
settime = 5e5 # years of accretion to simulate

if not bothchoice:
    if masschoice: # Let the final mass decide how many years of accretion to simulate
        settime = formationtime*np.log(stoppingmass/massscaling)/np.log(1/M0)
    elif timechoice: # Let the years of accretion to simulate decide what the stopping mass should be
        stoppingmass = massscaling*np.exp(settime/formationtime*np.log(1/M0))
print(str(sys.argv[0])+'. Starting mass: '+str(massscaling)+'. M0: '+"{:.2e}".format(M0)+'. Mfinal: '+str(stoppingmass)+'. Tfinal: '+"{:.2e}".format(settime))

# -------------------------------General functions-----------------------------
def delta(array): # calculates the forward difference to the next element in array. returns array with one less element than input array
    return array[1:]-array[:-1]

# calculates the mean value between subsequent elements in array. returns array with one less element than input array
def centeraveraging(array):
    return (array[1:]+array[:-1])/2

# initializing
volatiles = np.array(['hydrogen', 'carbon', 'nitrogen'])
gasvolatiles = np.array(['H2O', 'CO2', 'N2'])
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
sigmaBoltzmann = 1781.83355 # Boltzmann constant sigma = 5.67*1e-8 W/m^2 K^-4 converted to Petagram year^-3 K^-4
kB = 1.37490492e-26 # kB Boltzmann constant = 1.37490492e-23 J/K in Pg km2 yr-2 K-1

vmet = 1e-3*secondsperyear  # 1 m/s in km yr-1, falling speed of metal blobs
silicatedensity = 3.4 # Pg km-3
# 5.7 # Pg km-3, Earth's Core chapter by David Loper. PLACEHOLDER
coremetaldensity = 10.48
earthmass = 5.9712e12  # Earth's mass in Pg
venusmass = 4.86732e12 # Venus' mass in Pg
mproton = 1.67262192e-39 # proton mass in Pg

Lsun = 1.20862995e31 # solar luminosity. 3.84600e26 J/s in Pg km2 yr-3
AU = 149597871 # 1 AU in km  
a0 = 1.6*AU # CHOICE of starting point. 
a = np.copy(a0) # PLACEHOLDER. Use proper a(M(t)) growth track. 

# Calculate initial pebble mass accretion rate
if constantpebblerate:
    pebblerate0 = 1*earthmass/formationtime # Pg yr-1, PLACEHOLDER VALUE, x order(s) of magnitude less than what is required to form a 1 Earth mass planet over 5 Myr.
else:
    tau = formationtime/np.log(1/M0)
    pebblerate0 = massscaling*earthmass/tau # Pg yr-1

pebblemetalfrac = 0.3  # mass fraction of metal (iron) in accreted pebbles. Mass fraction here means met/(met+sil) PLACEHOLDER
coremassfraction = pebblemetalfrac # consistent with formation
# coreradiusfraction = 1/(1+coremetaldensity/silicatedensity*(1/coremassfraction-1))**(1/3) # 3480/6371 in Earth's Core chapter by David Loper. PLACEHOLDER

# Set planet mass, initial uncompressed radius is determined from CHOICE of planet mass
protomass = massscaling*earthmass
mc0 = coremassfraction*protomass # fraction of the total protoplanetary mass that is composed of the core
rcore0 = (3*mc0/(4*pi*coremetaldensity))**(1/3)
rsurf0 = ((protomass-mc0)/(4*pi*silicatedensity/3)+rcore0**3)**(1/3) # km, initial planet radius
        
# compression and expansion values
alpha0 = 3e-5  # volumetric thermal expansion coefficient, K-1
heatconductivity = 4*31425635.8 # base heat conductivity in AnatomyI, 4 W m-1 K-1 in Pg km year-3 K-1
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
Matmguess = Matmhigh # CHOICE, should it be? 
opacitypho = 1e-3*1e6  # (gray) Rosseland mean opacity from Badescu2010. 1e6 factor for conversion from m^2 kg^-1 to km^2 Pg^-1, PLACEHOLDER
def opacityfunc(p, T): # scaling from Rogers2024. pressure input in GPa
    pinbar = p*GPatobar
    alpha, beta = 0.68, 0.45
    kappa = 1.29e-2*1e6*(pinbar/1)**alpha*(1/1000)**beta # 1e5 factor for conversion from cm^2 g^-1 to km^2 Pg^-1    
    return kappa

monoatomicgamma = 5/3 # 1.666666
diatomicgamma = 7/5 # 1.4
triatomicgamma = 8/6 # 1.3333
lowgamma = 1.25 # higher temperatures --> lower Cp, lower Cp/Cv. 
envgamma = triatomicgamma # CHOICE. adiabatic gamma from the top of the outgassed atmosphere (bottom of the envelope) up to the RCB. 

#------------------------------Structure functions-----------------------------
# calculates volume of all magma ocean layers using layer boundaries r
def volumefunc(r): # calculates volume of all magma ocean layers using layer boundaries r
    return 4*pi*delta(r**3)/3


# calculates mass enclosed below each layer boundary
def menc(masses, mc = mc0):  
    return np.cumsum(np.concatenate(([0], masses)))+mc


# calculates the centerpoint density between density calculatesd at boundaries.
def centerdensity(densityatboundaries, r):
    rcenters = r[:-1]+delta(r)/2
    # interpolate the shell density then divide by r^2 to get density.
    return np.interp(rcenters, r, r**2*densityatboundaries)/rcenters**2


# calculates atmospheric mass from surface pressure and planet mass
def Matmfunc(psurf, rsurf, protomass=protomass):
    gsurf = G*protomass/rsurf**2
    return 4*pi*rsurf**2/gsurf*(psurf/GPaconversion)


# calculates surface pressure from masses and the surface radius
def psurffunc(Matm, rsurf, protomass=protomass):
    gsurf = G*protomass/rsurf**2
    return gsurf*Matm/(4*pi*rsurf**2)*GPaconversion


# # LEGACY WRONG. calculates the surface temperature assuming an adiabatic temperature profile from the photosphere down to the surface
# def pTsurf(rsurf, Matm=Matmguess, pebblerate=pebblerate, protomass=protomass, opacitypho=opacitypho, gamma=envgamma):
#     gsurf = G*protomass/rsurf**2
#     Ltot = pebblerate*G*protomass/rsurf #+accretionluminositysingle(pebblerate*pebblemetalfrac, rs, drs, Menc0) # full pebble luminosity 
#     psurf = gsurf*Matm/(4*pi*rsurf**2)*GPaconversion # GPa, pressure at surface
#     ppho = 2/3*gsurf/opacitypho*GPaconversion
#     Tpho = (Ltot/(4*pi*rsurf**2*sigmaBoltzmann))**(1/4) # temperature at the base of the photosphere
#     Tsurf = Tpho*(psurf/ppho)**((gamma-1)/gamma)
#     return psurf, Tsurf

# protoplanetary disk midplane pressure and temperature
def pTdisk(a=a0, Lstar=Lsun):
    phi = 0.05 # disk inclination angle 
    Tdisk = (Lstar*phi/(8*pi*sigmaBoltzmann*a**2))**(1/4)
    mgas = 2.2*mproton
    sigmagas = 0.017*(a/AU)**(-3/2) # column density of gas in disk. 1700 g/cm^2 in Pg/km^2
    kepfreq = 1*(a/AU)**(-3/2) # orbital frequency at a. 1 yr-1 at 1 AU. Kepler's third law.   
    cs = (kB*Tdisk/mgas)**(1/2) # sound speed in km/year
    pdisk = sigmagas*cs*kepfreq/(2*pi)**(1/2)*GPaconversion
    return pdisk, Tdisk

# calculates the pressure and temperature at the boundary between the exterior radiative region and the interior convective region of the atmosphere. PisoYoudin2014
def pTrcb(rsurf, pebblerate=pebblerate0, protomass=protomass, gamma=envgamma, a=a0, Lstar=Lsun, beta=2):
    pdisk, Tdisk = pTdisk(a, Lstar) # p, T atmospheric boundary conditions are that of the surrounding protoplanetary disk
    nabla_ad = (gamma-1)/gamma # adiabatic temperature gradient 
    nabla_inf = 1/(4-beta) # temperature gradient as p,T --> infinity
    dustopacity = 2*(Tdisk/100)**beta*1e5 # 2 cm2 g-1 at T = 100 K in km2 Pg-1 
    Ltot = pebblerate*G*protomass/rsurf 
    nabla_disk = 3*dustopacity*(pdisk/GPaconversion)*Ltot/(64*pi*G*protomass*sigmaBoltzmann*Tdisk**4)
    prcb = pdisk*(nabla_ad/nabla_disk-nabla_ad/nabla_inf)/(1-nabla_ad/nabla_inf)
    prcb = max(prcb, pdisk) # if prcb<pdisk then it is convective all the way from disk to surface.
    Trcb = Tdisk*(1+nabla_disk/nabla_inf*(prcb/pdisk-1))**(1/(4-beta))
    return prcb, Trcb # if prcb<pdisk then it is convective all the way from disk to surface.


# calculates pressure and temperature at the bottom of the envelope. 
def pTenvbot(rsurf, pebblerate=pebblerate0, protomass=protomass, gamma=envgamma, a=a0, Lstar=Lsun, beta=2):
    mgas = 2.2*mproton
    nabla_ad = (gamma-1)/gamma
    pdisk, Tdisk = pTdisk(a, Lstar)
    prcb, Trcb = pTrcb(rsurf, pebblerate, protomass, gamma, a, Lstar, beta)
    RBperRrcb = nabla_ad/(Trcb/Tdisk)*np.log(prcb/pdisk)
    # Tenv2 = Trcb+nabla_ad*mgas*G*protomass/kB*1/rsurf
    Tenv = Trcb+nabla_ad*mgas*G*protomass/kB*1/rsurf-Trcb*RBperRrcb # T_disk*R_B'/R_rcb term neglected. We assume R_B' is small since our thick atmosphere is not extended.
    penv = prcb*(Tenv/Trcb)**(1/nabla_ad) # adiabat w/ ideal gas law. 
    return penv, Tenv


def Tatmintegral(psurf, penvbot, Tenvbot, species='CO2', Miatm=None): # integrates the temperature from the bottom of the envelope to the surface of the planet
    dT = 5 # K, choice
    Tintegrand = Tenvbot+dT
    sum = 0 
    sumtarget = np.log(psurf/penvbot) # integral should sum to this quotient
    while sum < sumtarget:
        if species == 'mix':
            sum += dT*1/(Tintegrand*gasmixnabla(Tintegrand, Miatm))
        else:
            sum += dT*1/(Tintegrand*nabla_ad_atm(Tintegrand, species)) # rectangular integral approx. 
        Tintegrand += dT
    Tintegrand = max(Tenvbot, Tintegrand-dT/2) # take away half a step size and also make sure that it is bigger than the bottom envelope temperature
    return Tintegrand


def pTsurf(rsurf, Matm=Matmguess, pebblerate=pebblerate0, protomass=protomass, gamma=envgamma, a=a0, Lstar=Lsun, species='CO2', Miatm=None):
    gsurf = G*protomass/rsurf**2
    Ltot = pebblerate*G*protomass/rsurf
    penv, Tenv = pTenvbot(rsurf, pebblerate, protomass, gamma, a, Lstar)
    psurf = penv + gsurf*Matm/(4*pi*rsurf**2)*GPaconversion # thin outgassed atmosphere approximation
    # Tsurf = Tenv*(psurf/penv)**((gamma-1)/gamma) # DOESN'T WORK. Would work if Cp(T) was constant with T. 
    Tatmbot = Tatmintegral(psurf, penv, Tenv, species, Miatm)
    Tsurf = (Tatmbot**4+Ltot/(4*pi*rsurf**2*sigmaBoltzmann))**(1/4)
    return psurf, Tsurf


# K-1, calculates the volumetric thermal expansion coefficient as a function in units of GPa of pressure. Abe1997
def thermalexpansion(pressure):
    return alpha0*(pressure*K0prime/K0+1)**((1-K0prime)/K0prime)


# Shomate equation for specific heat capacity of silicate as function of temperature. NIST Webbook, Condensed phase thermochemistry data
def Cpfuncsil(temperature, material='e'):  
    Tper1000 = temperature/1000  # Temperature/(1000 K)
    if material == 'e':  # enstatite, MgSiO3
        molarmass = 100.3887/1000  # kg/mol
        a1, b1, c1, d1, e1 = 146.4400, -1.499926e-7, 6.220145e-8, - \
            8.733222e-9, -3.144171e-8  # 1850 K < T
        a2, b2, c2, d2, e2 = 37.72742, 110.1852, - \
            50.79836, 7.968637, 15.16081  # 903 K < T < 1850 K
        Cpermole = np.where(temperature > 1850, a1+b1*Tper1000+c1*Tper1000**2+d1*Tper1000**3+e1/Tper1000 **
                            2, a2+b2*Tper1000+c2*Tper1000**2+d2*Tper1000**3+e2/Tper1000**2)  # Shomate equation, J/(mol*K))
    elif material == 'f':  # forsterite, Mg2Si04
        molarmass = 140.6931/1000  # kg/mol
        a1, b1, c1, d1, e1 = 205.0164, -2.196956e-7, 6.630888e-8, - \
            6.834356e-9, -1.298099e-7  # 2171 K < T
        a2, b2, c2, d2, e2 = 140.8749, 47.15201, - \
            12.22770, 1.721771, -3.147210  # T < 2171 K
        Cpermole = np.where(temperature > 2171, a1+b1*Tper1000+c1*Tper1000**2+d1*Tper1000**3+e1/Tper1000**2,
                             a2+b2*Tper1000+c2*Tper1000**2+d2*Tper1000**3+e2/Tper1000**2)  # Shomate equation, J/(mol*K))
    else:
        sys.exit('No silicate material matches input')
    return Cpermole/molarmass*Cpconv  # units of km2 year-2 K-1

# Calculates the adiabatic nabla for a given temperature in the outgassed atmosphere of a specified species.
# Uses Shomate equation for specific heat capacity of material as function of temperature. NIST Webbook, gas phase thermochemistry data
def nabla_ad_atm(temperature, species='CO2', returnCp=False): 
    Tper1000 = temperature/1000
    R = 8.31446261815324 # Boltzmann molar gas constant in J K-1 mol-1
    if species=='H2O':
        molarmass = 0.01801528 # kg/mol 
        Rspecific = R/molarmass # J kg-1 K-1
        a1, b1, c1, d1, e1 = 30.09200, 6.832514, 6.793435, -2.53448, 0.082139
        a2, b2, c2, d2, e2 = 41.96426, 8.622053, -1.499780, 0.098119, -11.15764
        Cpermole = np.where(temperature < 1700, a1+b1*Tper1000+c1*Tper1000**2+d1*Tper1000**3+e1/Tper1000**2,
                             a2+b2*Tper1000+c2*Tper1000**2+d2*Tper1000**3+e2/Tper1000**2)  # Shomate equation, J/(mol*K))
    elif species=='CO2':
        molarmass = 0.04401 # kg/mol
        Rspecific = R/molarmass # J kg-1 K-1
        a1, b1, c1, d1, e1 = 24.99735, 55.18696, -33.69137, 7.948387, -0.136638
        a2, b2, c2, d2, e2 = 58.16639, 2.720074, -0.492289, 0.038844, -6.447293
        Cpermole = np.where(temperature < 1200, a1+b1*Tper1000+c1*Tper1000**2+d1*Tper1000**3+e1/Tper1000**2,
                             a2+b2*Tper1000+c2*Tper1000**2+d2*Tper1000**3+e2/Tper1000**2)  # Shomate equation, J/(mol*K))
    elif species=='N2':
        molarmass = 0.028014 # kg/mol 
        Rspecific = R/molarmass # J kg-1 K-1
        a1, b1, c1, d1, e1 = 19.50583, 19.88705, -8.598535, 1.369784, 0.527601
        a2, b2, c2, d2, e2 = 35.51872, 1.128728, -0.196103, 0.014662, -4.553760
        Cpermole = np.where(temperature<2000, a1+b1*Tper1000+c1*Tper1000**2+d1*Tper1000**3+e1/Tper1000**2,
                             a2+b2*Tper1000+c2*Tper1000**2+d2*Tper1000**3+e2/Tper1000**2)  # Shomate equation, J/(mol*K))
    else: 
        sys.exit('No gas species matches input')
    Cp = Cpermole/molarmass # J/(kg*K)
    if returnCp:
        return Cp # return Cp of a single species if the atmosphere is a mix
    else: 
        return Rspecific/Cp # return nabla_ad of a single gas species. unitless

def gasmixnabla(temperature, Miatm):
    Cparray = [nabla_ad_atm(temperature, species=i, returnCp=True) for i in gasvolatiles]
    Cpmean = np.sum(Cparray*Miatm/np.sum(Miatm)) # multiply Cp by corresponding gas mass fraction and sum
    molarmassarray = np.array([0.01801528, 0.04401, 0.028014]) # kg/mol
    massfractions = Miatm/np.sum(Miatm) 
    meanmolarmass = 1/np.sum(massfractions/molarmassarray) # calculate the mean molar mass
    R = 8.31446261815324 # Boltzmann molar gas constant in J K-1 mol-1
    Rspecific = R/meanmolarmass # Boltzmann gas constant for our specific mixture
    return Rspecific/Cpmean # return nabla_ad of a gas mixture


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
        Cp = Cpfuncsil(adiabattemperatureflip[i-1])
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
    densityflip[0] = topdensity # density if not subject to bulk compression/thermal expansion
    for i in range(1, number_of_structure_boundaries):
        densityflip[i] = densityflip[i-1]*((K0+pressureflip[i]*K0prime)/(K0+pressureflip[i-1]*K0prime))**(
            1/K0prime)*np.exp(-alphaflip[i-1]*(temperatureflip[i]-temperatureflip[i-1]))
        # print('Temperature expansion: '+str(np.exp(-alphaflip[i-1]*(temperatureflip[i]-temperatureflip[i-1]))))
        # print('Pressure compression: '+str(((K0+pressureflip[i]*K0prime)/(K0+pressureflip[i-1]*K0prime))**(
        #     1/K0prime)))
        # print('Their product: '+str(densityflip[i]/densityflip[i-1]))
    return np.flip(densityflip)


# calculates new layerboundaries to keep mass inside each layer constant despite new densities
def newlayerboundaries(layermasses, newdensity, rcore=rcore0, number_of_structure_boundaries = number_of_structure_boundaries):
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
    Cp = Cpfuncsil(temperature)  # calculates specific heat capacity
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
        # return temperature, adiabattemperature, Ra
        return adiabattemperature, adiabattemperature, Ra # PLACEHOLDER. Should follow the adiabat perfectly given how convective the MO is. 
    else:
        return temperature


# calculates the Nusselt number from the Rayleigh number in soft or hard turbulence regime
def NufromRa(Ra):
    Rachard = 200  # critical Rayleigh number for hard turbulence regime, AnatomyI
    Racweak = 1000
    # calculates the Nusselt Number depending on turbulence regime
    Nu = np.where(Ra > 1e19, (Ra/Rachard)**(2/7), (Ra/Racweak)**(1/3))
    return Nu+1  # +1 to account for heat conduction still occurring even if there were no convection


# calculates the diffusivity from the Nusselt number, 
def DfromNu(Nu, density, temperature, K=heatconductivity): 
    D = np.mean(Nu*K/(density*Cpfuncsil(temperature)))
    return D


# interpolate the density from the new layer centers to the linearly spaced layer centers, then calculate the new layermasses
def massanddensityinterpolation(r, rnew, newdensity, totalmass):
    # center of layers, where we want to know the density to calculate mass from
    rcenters = centeraveraging(r)
    # linearly interpolate w.r.t shell density such that total mass is preserved in interpolation
    density = np.interp(rcenters, rnew, rnew**2*newdensity)/rcenters**2
    layermasses = density*volumefunc(r)
    layermasses *= np.sum(totalmass)/np.sum(layermasses) # scale to make absolutely sure that the silicate mass is conserved with interpolation. DENSITY IS STILL UNSCALED
    density = layermasses/volumefunc(r) # fix density to be consistent with layermasses
    return layermasses, density

    
# calculates p, T, rho profile and Ra when rho is known for the layers
def restructuringfromdensity(r, rho, Matm, mc, pebblerate = pebblerate0, number_of_structure_boundaries = number_of_structure_boundaries, iterations = number_of_structure_iterations, Miatm=None, species='CO2', plotting = False):
    M = rho*volumefunc(r)  # calculates initial layer mass
    return restructuring(r, M, Matm, mc, pebblerate, number_of_structure_boundaries, iterations, Miatm, species, plotting)
    

# calculates p, T, rho profile and Ra when the layermasses are known 
def restructuring(r, M, Matm, mc, pebblerate = pebblerate0, number_of_structure_boundaries = number_of_structure_boundaries, iterations = number_of_structure_iterations, Miatm=None, species='CO2', plotting=False):
    starttime = time.time()
    rvar = np.linspace(r[0], r[-1], number_of_structure_boundaries) # create refined boundaries between layers for the structure calculations
    drvar = delta(rvar)
    
    rho = M/volumefunc(r) # calculates the densities in the old layers
    initialdensity = np.interp(rvar, centeraveraging(r), rho) # interpolate to the new layers
    
    layermasses = centeraveraging(initialdensity)*volumefunc(rvar) # remains constant, only layer boundaries will shift with iterations
    layermasses *= np.sum(M)/np.sum(layermasses)
    Menc = menc(layermasses, mc) # enclosed mass. remains constant since mass inside each layer remains constant, only the layer thicknessses change with iteration
    print('M: '+str((np.sum(M)+mc)/earthmass)) # PLACEHOLDER
    print('layermasses: '+ str(np.sum(layermasses)/earthmass)) # PLACEHOLDER
    
    psurf, Tsurf = pTsurf(rvar[-1], Matm, pebblerate, protomass, a=a, Miatm=Miatm, species=species) # calculate pressure and temperature at the surface of the world
    pressure = pressurefunc(rvar, drvar, initialdensity, Menc, psurf) # calculates initial pressure
    temperature = topdownadiabat(rvar, drvar, pressure, Menc, Tsurf) # calculates temperature
    
    for j in range(iterations):
        density = densityfunc(pressure, temperature) # implicit 
        rvar = newlayerboundaries(layermasses, density, r[0]) # set core radius to innermost layerboundary of input r
        drvar = delta(rvar)    

        rmiddle = 1/2*(rvar[-1]-rvar[0]) # center of magma ocean, use as the position where the accretion luminosity is supplied
        Lmet = accretionluminositysingle(pebblerate*pebblemetalfrac, rmiddle, rvar, drvar, Menc) # calculates the energy deposited into the magma ocean by the accretion of the metals in the pebbles 
        
        psurf, Tsurf = pTsurf(rvar[-1], Matm, pebblerate, protomass, a=a, Miatm=Miatm, species=species) # calculate pressure and temperature at the surface of the world
        pressure = pressurefunc(rvar, drvar, density, Menc, psurf)
        temperature, adiabattemperature, Ra = temperaturefunc(rvar, drvar, rmiddle, pressure, temperature, density, Menc, Tsurf, Lmet, heatconductivity, True)
        
        if (j % structureplotstepspacing == 0 or j == number_of_structure_iterations-1) and plotting:
            Mtotstr = "{:.2f}".format(Menc[-1]/earthmass)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,4))
            fig.suptitle('Structure of magma ocean after '+str(j+1) +
                         ' time steps. Total mass: '+Mtotstr+' M'+'$_{E}$')

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

            # ax1.locator_params(axis='both', nbins=6)  # set number of tick-marks
            # ax2.locator_params(axis='both', nbins=6)
            ax1.grid(True)  # plot a grid
            ax2.grid(True)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped

    rshort = np.linspace(rvar[0], rvar[-1], len(r))
    rcentershort = centeraveraging(rshort)
    pressureshort = np.interp(rcentershort, rvar, pressure) 
    temperatureshort = np.interp(rcentershort, rvar, temperature)
    massshort, densityshort = massanddensityinterpolation(rshort, rvar, density, np.sum(layermasses))
    print('massshort: '+str(np.sum(massshort/earthmass))) # PLACEHOLDER
    print('Rcmb: '+str(rshort[0])+' km, Rsurf: '+str(rshort[-1])+' km, Tsurf: '+str(temperatureshort[-1])+' K, Psurf: '+str(psurf*GPatobar)+' bar') # PLACEHOLDER
    D = DfromNu(NufromRa(Ra), densityshort, temperatureshort)
        
    compressioncalctime = time.time()-starttime
    return rshort, psurf, Tsurf, pressureshort, temperatureshort, densityshort, D, compressioncalctime

#---------------------Initial Structure without volatiles----------------------
# create magma ocean layer boundaries and calculate (p, T, rho) at the center of these layers
rk, dr0 = np.linspace(rcore0, rsurf0, number_of_boundaries, retstep=True) # spatial discretization to be used for volatile transport
initialvolumes = volumefunc(rk) # calculate volume of each magma ocean layer 
initialMmet = pebblerate0*pebblemetalfrac*dr0/vmet # metal inside each layer, LET PEBBLERATE CHANGE?
initialMtot = initialMmet+initialvolumes*silicatedensity
initialdensity = initialMtot/initialvolumes

# compressed structure
rk, psurf, Tsurf, pressure, temperature, density, D, compressioncalctime = restructuringfromdensity(rk, initialdensity, Matmguess, mc0, pebblerate0, number_of_structure_boundaries, number_of_structure_iterations, plotting=True)
dr = rk[1]-rk[0] # same layerwidth for all layers
rsurf = rk[-1]  # new surface is the compressed surface
layercenters = centeraveraging(rk)
layervolumes = volumefunc(rk) # calculate volume of each magma ocean layer 
Mmet = pebblerate0*pebblemetalfrac*dr/vmet # metal inside each layer, LET PEBBLERATE CHANGE?
Msil = density*layervolumes # compressed density*volume is mass of silicates
Msilmet = Msil+Mmet # mass of both sil and met 


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
    Tperformula = 1 # one Mg in MgSiO3
else: # forsterite, Mg2SiO4
    amuMelt = 140
    Tperformula = 2 # two Mg in MgSiO3

# number of volatile-bearing products per volatile reactant
alphaH2O = 2 # H2O(gas) + O(melt) = 2 OH(melt) e.g. Stolper1982. 
alphaCO2 = 1 # CO2(gas) + O(melt) = CO3(melt) e.g. Ortenzi2016, Fegley2020. Or more accurately M2SiO4(melt)+CO2(g) = MCO3(melt) + MSiO3(melt) e.g. Spera1980. 
alphaN2chem = 2 # N2(gas)+_*H2O(gas)+(3-_)*O(melt) = 2 NH_ + 3/2 O(gas) e.g. Grewal2020.
alphaN2phys = 1 # N2(gas) = N2(melt)  
# alphai = np.array([alphaH2O, alphaCO2, alphaN2]) 
alphai = np.array([[alphaH2O, 1],[alphaCO2, 1],[alphaN2chem, alphaN2phys]]) # 1 for dummy variables where Ki = 0. 

# initial mass of volatiles in melt, CHOICE
XH2O = 2e3*1e-6 # 2e3 ppmw, AnatomyIII
XC = 600*1e-6 # 600 ppmw total estimated from Anders2021 or AnatomyIII. 27 ppmw of total planet mass of Venus lies in atmosphere (google) 
XN = 10**0.5*1e-6 # mass fraction of total planet mass of Venus in atmosphere, from same figure in AnatomyIII
XH = XH2O*2/18 # PLACEHOLDER DOES THIS EVEN MAKE SENSE, 2 of the 18 amu in H2O is from the hydrogen
XFeO = 0.1 # 0.3*(55.845+16)/55.845 # PLACEHOLDER DOES THIS EVEN MAKE SENSE. 30% of bulk mass is metal. 
Xi = np.array([XH2O, XC, XN]) # array to store the volatiles in.

# initial masses each volatile element in the silicate melt
MH2O = XH2O*Msil
MC = XC*Msil
MN = XN*Msil
Mvol0 = np.array([MH2O, MC, MN])
Mtot0 = Msil+Mmet+np.sum(Mvol0, axis=0)
volfrac0 = Mvol0/Mtot0 # fraction of volatiles in each magma ocean layer at the start 
Msilvol = Msil+np.sum(Mvol0, axis=0) # mass of volatiles + silicate melt inside each layer
Micore0 = np.zeros(len(volatiles)) # start with no volatiles in the core

#-----------------------------Parametrized functions---------------------------
def IWfO2func(p, T): # oxygen fugacity of the IW-buffer Fe + O <-> FeO. Hirschmann2021. Valid for T>1000 K and P between 1e-4 and 100 GPa. "It extrapolates smoothly to higher temperature, though not calibrated above 3000 K." 
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

def fO2func(p, T, deltaIW=-2.2, cappedfraction=False): # oxygen fugacity 
    fO2para = IWfO2func(p, T)*10**deltaIW # fO2 as defined by Hirschmann2021 parametrization
    if cappedfraction:
        fO2max = p*cappedfraction # fO2 cannot exceed some fraction of the total pressure
        return np.where(fO2max<fO2para, fO2max, fO2para)
    else: 
        return fO2para
    
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
    XH2Omet = np.where(XH2Omet>1e-8, XH2Omet, 1e-8) # PREVENTS log10(0). 
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
fO2 = fO2func(pressure, temperature) 
fO2surf = fO2[-1]

# plot oxygen fugacity as function of p, T. Add our own p, T curve. 
X, Y = np.meshgrid(pressure, temperature)
Z = np.array([np.log10(fO2func(p,t)) for p in pressure for t in temperature]).reshape(number_of_layers, number_of_layers)
fO2fig = plt.figure()
CS = plt.contourf(X,Y,Z,20)
cbar = fO2fig.colorbar(CS)
plt.plot(pressure, temperature, color='k') #, norm=pltcolors.TwoSlopeNorm(0), cmap='bwr' can be used
# invert axes so that going to the left or going down on the plot corresponds to further into the magma ocean
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.title('log Oxygen fugacity at deltaIW') # PLACEHOLDER, specify what deltaIW is in the figure title
plt.xlabel('Pressure [GPa]')
plt.ylabel('Temperature [K]')
plt.show()

# estimate initial partition coefficients using bulk mass fractions globally
if evendist: 
    PH2O, CH2Omet = iterativePH2O(pressure, Mvol0[0], Msil)
    PC = PCfunc(pressure, temperature, 0)
    PN = PNfunc(pressure, temperature, XFeO)
    Pvol = np.array([PH2O, PC, PN]) # array of all the volatile species' partition coefficients (at all magma ocean layers)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,4))
    fig.suptitle('Metal droplets/silicate melt partition coefficient of volatile species')
    ax1.set_prop_cycle('color', vcolors)
    ax1.semilogy(layercenters, np.transpose(Pvol), label=volatiles)
    ax1.set_xlabel('radius [km]')
    ax1.set_ylabel('Partition coefficient')
    ax1.legend()

    ax2.set_prop_cycle('color', vcolors)
    ax2.semilogy(pressure, np.transpose(Pvol), label=volatiles)
    ax2.set_xlabel('Pressure [GPa]')

#-----------------------Gas-melt interaction functions-------------------------

# pressure/Matm-dependent equilibrium coefficient function array
def Kifunc(Matm, fO2surf, rsurf=rsurf, protomass=protomass, pebblerate=pebblerate0, H2Oindex=0, meltindex=0, Miatm=None, species='CO2',): # choice of melt from which experimental K for CO2 is taken
    psurf, Tsurf = pTsurf(rsurf, Matm=Matm, protomass=protomass, pebblerate=pebblerate, a=a, Miatm=Miatm, species=species)
    pinkbar = psurf*GPatokbar
    pinbar = psurf*GPatobar
    
    KH2Oburnham = 1/(np.exp(5.00+np.log(pinbar)*(4.481e-8*Tsurf**2-1.51e-4*Tsurf-1.137)+np.log(pinbar)**2*(1.831e-8*Tsurf**2-4.882e-5*Tsurf+4.656e-2)+7.80e-3*np.log(pinbar)**3-5.012e-4*np.log(pinbar)**4+Tsurf*(4.754e-3-1.621e-6*Tsurf))) # Burnham 1994, eq. 7. Taken to power -1 because their K is defined for the inverse reaction. 
    KH2Ostolper = 0.2*np.exp(-3+pinkbar/10) # Stolper1982, fig. 4 and eq. 8. Product of K1 and K2. Slopes up one log unit after 10 kbar.
    KH2O = np.array([KH2Oburnham, KH2Ostolper])[H2Oindex] # index choice
    
    KCO2withNepheline = 1/np.exp(12.5+5.5*pinkbar/30) # Spera1980, fig. 1. Minus sign since we want the reverse equilibrium constant
    KCO2withOlivineMelilite = 1/np.exp(13+5*pinkbar/30) # Spera1980, fig. 3. -||-
    KCO2 = np.array([KCO2withNepheline, KCO2withOlivineMelilite])[meltindex] # index choice
    
    R = 8.31446261815324 # J/(mol*K), universal gas constant
    KN2chem = np.exp(-(183733+172*Tsurf-5*pinbar)/(R*Tsurf))*fO2surf**(-3/4) # Bernadou2021, eq. 14 & Table 6. fO2 dependence from eq. 18
    KN2phys = np.exp(-(29344+121*Tsurf+4*pinbar)/(R*Tsurf)) # Bernadou2021, eq. 13 & Table 6.
    
    return np.array([[KH2O, 0], [KCO2, 0], [KN2chem, KN2phys]])


# returns masses of each volatile in the first atmosphere to be in equilibrium with the assumed starting volatile fraction in the silicate melt of the top layer of the magma ocean
def startingatmosphere(Matm, Xiktop, AMUi, alphai, fO2surf):
    Ki = Kifunc(Matm, fO2surf)
    Kiused = np.max(Ki, axis=1) # PLACEHOLDER 
    frac = Xiktop**alphai[:,0]*AMUi/Kiused
    Miatm0 = frac/np.sum(frac)*Matm
    return Miatm0


# input needs to be arrays, sensitive to actual values of input parameters (must be fairly close to equilibrium already) be careful
def Miatmeqfunc(Miatmguess, rsurf, AMUi, alphai, Msiltop, Misil0, fO2surf, protomass=protomass, pebblerate=pebblerate0, method='fsolve'): # function to find the dissolution equilbrium solution of the mass of volatile species in the atmosphere 
    Mitot0 = Miatmguess+Misil0 # sum of volatiles in either reservoir before equilibriation
    def Miatmrootfunc(Miatm): # function of Miatm that evaluates to 0 at physical+chemical dissolution equilbrium between top layer of magma ocean and the atmosphere
        frac = Miatm/AMUi
        Ki = Kifunc(np.sum(Miatm), fO2surf, rsurf, protomass, pebblerate, Miatm=Miatm, species='mix') # calculate the equilibrium constants from pressure (or Matm)
        return 1/Mitot0*(Miatm+((Ki[:,0]*frac/np.sum(frac))**(1/alphai[:,0])+(Ki[:,1]*frac/np.sum(frac))**(1/alphai[:,1]))*Msiltop-Mitot0)
    # def derivativefunc(Miatm): 
        # quit # write fprime = derivativefunc as argument to fsolve
    if method=='fsolve': # appears to be stably solving further-from-equilibrium starting guesses
        Miatmeq = fsolve(Miatmrootfunc, Miatmguess, xtol=1e-8) # find roots to the function with a starting guess being the unequilibrated atmospheric mass of the volatile species.
    else:
        Miatmeq = broyden1(Miatmrootfunc, Miatmguess, f_tol=1e-8) # use Broyden's good method instead
    print(str(np.isclose(Miatmrootfunc(Miatmeq), np.zeros_like(Miatmeq)))+'. Miatm,eq/Miatm0: '+str(Miatmeq/Miatmguess)) # check if roots are valid

    Misileq = Mitot0-Miatmeq # ERROR: CAN PRODUCE NEGATIVE TOP MASSES. mass of volatile in magma ocean = total volatile mass minus volatile mass in atmosphere
    meltmassfraci = massfraction(Misileq, Msiltop) # mass fraction of volatile in top layer of magma ocean
    atmmassfraci = Miatmeq/np.sum(Miatmeq) # mass fraction of volatile in the atmosphere
    psurf = psurffunc(np.sum(Miatmeq), rsurf)
    return Miatmeq, Misileq, meltmassfraci, atmmassfraci, psurf, Miatmeq/Miatmguess


Miatmguess = startingatmosphere(Matmguess, Mvol0[:,-1]/Msilvol[-1], amui, alphai, fO2surf)
eqstate = Miatmeqfunc(Miatmguess, rsurf, amui, alphai, Msil[-1], Mvol0[:,-1], fO2surf, protomass, pebblerate0) # state of mass conservation and dissolution equilibrium
Miatm0 = eqstate[0]
Matm0 = np.sum(Miatm0)
print(Matm0/np.sum(Miatmguess))

#-------------------convection (diffusion) and sedimentation-------------------

#----------------------------------Transport-----------------------------------
stdmax = cspread*dr # km
dt = stdmax**2/(2*D) # diffusivity (calculated during restructuring step as enhanced conductivity to model convectivity) limits the timestep since volatile flux only goes to neighboring magma ocean layers
number_of_timesteps = int(settime/dt)  # compute the number of timesteps
plotstepspacing = max(plotstepspacing, int(number_of_timesteps/max_number_of_plots))
savingspacing = max(1, int(number_of_timesteps/max_len_savefiles))
print('daystocomplete = '+str(settime/dt/18/86400)) # approximately

# LEGACY SOLVER
# def tridiagsolver(rk, Mvol0, Msil, Mmet, vmet, D, Pvol, Micore0, dt=dt): # solves tri-diagonal matrix for volatile motion to neighboring layers of magma ocean via sedimentation and convection (modelled as enhanced diffusion) 
#    dr = rk[1]-rk[0] # constant layer width
#    rhosil = centeraveraging(Msil/volumefunc(rk)) # calculate density of silicate AT layer boundaries
#    
#    # Layers not adjacent to the CMB or the surface
#    layerabove = (4*pi*D*rhosil*rk[1:-1]**2+vmet*Pvol[:,1:]*Mmet)/(dr*(Msil[1:]+(Pvol[:,1:])*Mmet)) # mass flux from layer above (terms = 1 conv. down, 1 sed. down)
#    ownlayer = (-4*pi*D/dr*(rhosil[1:]*rk[2:-1]**2+rhosil[:-1]*rk[1:-2]**2)-vmet/dr*Pvol[:,1:-1]*Mmet)/(Msil[1:-1]+Pvol[:,1:-1]*Mmet) # mass flux away from layer (terms = 2 conv. (one up; one down), 1 sed. down)
#    layerbelow = 4*pi*D*rhosil*rk[1:-1]**2/(dr*(Msil[:-1]+Pvol[:,:-1])) # mass flux from layer below. (terms = 1 conv. from layer below)
#    # layerabove[:,1:]+ownlayer+layerbelow[:,:-1] # should be small
#    
#    bottomlayerloss = (-4*pi*D*rhosil[0]*rk[1]**2-vmet*Pvol[:,0]*Mmet)/(dr*(Msil[0]+Pvol[:,0]*Mmet))
#    toplayerloss = (-4*pi*D*rhosil[-1]*rk[-2]**2-vmet*Pvol[:,-1]*Mmet)/(dr*(Msil[-1]+Pvol[:,-1]*Mmet))
#    diag = np.array([np.append(np.concatenate(([bottomlayerloss[i]], ownlayer[i])), toplayerloss[i]) for i in irange]) # elements on the diagonal of the matrix A where dM_ik/dt = A M_ik for any volatile i. 
#    
#    Sik = Mvol0/(dr*(Msil+Pvol*Mmet)) # layer mass fraction of Mvol0 in silicate melt
#    
#    conv = np.ones_like(Mvol0)
#    conv[:,1:-1] *= 4*pi*D*(rhosil[1:]*rk[2:-1]**2*(Sik[:,2:]-Sik[:,1:-1])+rhosil[:-1]*rk[1:-2]**2*(Sik[:,:-2]-Sik[:,1:-1])) # two terms of convection (4 if you consider the discretization of dX/dr for each of the two terms)
#    conv[:,0] *= 4*pi*D*rhosil[0]*rk[1]**2*(Sik[:,1]-Sik[:,0])
#    conv[:,-1] *= 4*pi*D*rhosil[-1]*rk[-2]**2*(Sik[:,-2]-Sik[:,-1])
#    
#    sedi = np.ones_like(Mvol0)
#    sedi[:,1:-1] *= vmet*Mmet*(Sik[:,2:]*Pvol[:,2:]-Sik[:,1:-1]*Pvol[:,1:-1]) # two terms in general
#    sedi[:,0] *= vmet*Mmet*(Sik[:,1]*Pvol[:,1]-Sik[:,0]*Pvol[:,0]) # two terms at MO layer above CMB
#    sedi[:,-1] *= -vmet*Mmet*Sik[:,-1]*Pvol[:,-1] # one term at topmost MO layer
#
#    # extending the matrix to include the volatile mass in the core
#    coregain = vmet*Pvol[:,0]*Mmet/(dr*(Msil[0]+Pvol[:,0]*Mmet)) # coefficient for rate of mass gain for the core by sedimentation
#    layerabove = np.insert(layerabove, 0, coregain, axis=1) # add the coregain element for each of the volatiles (i.e. before element 0 in axis = 1)
#    diag = np.insert(diag, 0, 0, axis=1) # same but add 0 for each volatile 
#    layerbelow = np.insert(layerbelow, 0, 0, axis=1) # -||- 
#
#    Mvolpcore0 = np.insert(Mvol0, 0, Micore0, axis=1) # create volatile masses array that has core mass as first element
#    A = np.zeros_like(volatiles, dtype=object)
#    Mvolpcore = np.zeros_like(Mvolpcore0)
#    for i in irange:
#        A[i] = np.diag(layerbelow[i], -1) + np.diag(diag[i], 0) + np.diag(layerabove[i], 1) # creates the tridiagonal matrix
#        eigval, eigvec = eig(A[i]) # print(np.linalg.cond(eigvec)) # < 1e15 is good
#        Cs = np.linalg.solve(eigvec, Mvolpcore0[i]) # find coefficients to match expression of Mvol(t) to known Mvol0 = Mvol(0)
#        Mvolpcore[i,:] = np.real((Cs*np.exp(eigval*dt))@eigvec.transpose())
#    # coreloss = np.sum(Mvol, axis=1)-np.sum(Mvol0, axis=1) # calculates how much mass has gone into the core
#    Micore = Mvolpcore[:,0] # first element corresponds to core volatile mass inventory
#    Mvol = Mvolpcore[:,1:] # the rest correspond to each layer of the magma ocean
#    corediff = Micore-Micore0
#    return Mvol, conv, sedi, Micore, corediff


def eigandVinv(rk, Msil, Mmet, vmet, D, Pvol):
    dr = rk[1]-rk[0] # constant layer width
    rhosil = centeraveraging(Msil/volumefunc(rk)) # calculate density of silicate AT layer boundaries
    
    # Layers not adjacent to the CMB or the surface
    layerabove = (4*pi*D*rhosil*rk[1:-1]**2+vmet*Pvol[:,1:]*Mmet)/(dr*(Msil[1:]+(Pvol[:,1:])*Mmet)) # mass flux from layer above (terms = 1 conv. down, 1 sed. down)
    ownlayer = (-4*pi*D/dr*(rhosil[1:]*rk[2:-1]**2+rhosil[:-1]*rk[1:-2]**2)-vmet/dr*Pvol[:,1:-1]*Mmet)/(Msil[1:-1]+Pvol[:,1:-1]*Mmet) # mass flux away from layer (terms = 2 conv. (one up; one down), 1 sed. down)
    layerbelow = 4*pi*D*rhosil*rk[1:-1]**2/(dr*(Msil[:-1]+Pvol[:,:-1])) # mass flux from layer below. (terms = 1 conv. from layer below)
    
    bottomlayerloss = (-4*pi*D*rhosil[0]*rk[1]**2-vmet*Pvol[:,0]*Mmet)/(dr*(Msil[0]+Pvol[:,0]*Mmet))
    toplayerloss = (-4*pi*D*rhosil[-1]*rk[-2]**2-vmet*Pvol[:,-1]*Mmet)/(dr*(Msil[-1]+Pvol[:,-1]*Mmet))
    diag = np.array([np.append(np.concatenate(([bottomlayerloss[i]], ownlayer[i])), toplayerloss[i]) for i in irange]) # elements on the diagonal of the matrix A where dM_ik/dt = A M_ik for any volatile i. 
    
    # extending the matrix to include the volatile mass in the core
    if vmet != 0:
        coregain = vmet*Pvol[:,0]*Mmet/(dr*(Msil[0]+Pvol[:,0]*Mmet)) # coefficient for rate of mass gain for the core by sedimentation
        layerabove = np.insert(layerabove, 0, coregain, axis=1) # add the coregain element for each of the volatiles (i.e. before element 0 in axis = 1)
        diag = np.insert(diag, 0, 0, axis=1) # same but add 0 for each volatile 
        layerbelow = np.insert(layerbelow, 0, 0, axis=1) # -||- 

    A = np.zeros_like(volatiles, dtype=object)
    Vinv = np.zeros_like(volatiles, dtype=object)
    eigval = np.zeros_like(volatiles, dtype=object)
    eigvec = np.zeros_like(volatiles, dtype=object)
    for i in irange:
        A[i] = np.diag(layerbelow[i], -1) + np.diag(diag[i], 0) + np.diag(layerabove[i], 1) # creates the tridiagonal matrix
        eigval[i], eigvec[i] = eig(A[i]) # print(np.linalg.cond(eigvec)) # < 1e15 is good
        Vinv[i] = inv(eigvec[i])
    return eigval, eigvec, Vinv

def transMvol(Mvol0, Micore0, eigval, eigvec, Vinv, dt=dt, vmet=vmet):
    if vmet != 0: # check if we have sedimentation into core
        Mvol0 = np.insert(Mvol0, 0, Micore0, axis=1) # extend volatile masses array to have core mass as first element
    Mvol = np.zeros_like(Mvol0) # initialize Mvol(t)
    Cij = np.zeros_like(volatiles, dtype=object)
    for i in irange:
        Cij[i] = np.sum(Vinv[i]*Mvol0[i], axis=1) # IS THIS SUMMATION CORRECT? I think so. Equiv. to Vinv[i]@Mvolpcore0[i]
        Mvol[i] = np.real((Cij[i]*np.exp(eigval[i]*dt))@eigvec[i].transpose())
    if vmet != 0: # check if we have sedimentation into core
        Micore = Mvol[:,0] # first element corresponds to core volatile mass inventory
        corediff = Micore-Micore0
        Mvol = Mvol[:,1:] # the rest correspond to each layer of the magma ocean
    else:
        Micore = Micore0
        corediff = 0
    return Mvol, Micore, corediff

# Check that the transport method of eigandVinv and transMvol is working together
# eigval, eigvec, Vinv = eigandVinv(rk, Msil, Mmet, vmet, D, Pvol)
# Mvol, Micore, _ = transMvol(Mvol0, Micore0, eigval, eigvec, Vinv, dt, vmet)
# plt.figure()
# for i in irange:
#     plt.plot(layercenters, (Mvol[i]-Mvol0[i])/Mvol0[i], '.', color=vcolors[i], label=volatiles[i])
# plt.title('Volatile mass change in magma ocean')
# plt.xlabel('radius [km]')
# plt.ylabel('Change relative to previous volatile layer mass')
# plt.legend(loc='best')

protomass0 = mc0+np.sum(Micore0)+np.sum(Mtot0)+np.sum(Miatm0)+np.sum(Mvol0)
if nosedifrac>0: # set to 0 at start iff chosen. 
    vmet = 0
    Mmet=0
eigval0, eigvec0, Vinv0 = eigandVinv(rk, Msil, Mmet, vmet, D, Pvol) # calculate initial eigenvalues & eigenvectors of Ai matrix and the inverse of the matrix of these eigenvectors. 
# initialize the arrays of all time step variables
eigval, eigvec, Vinv = np.copy(eigval0), np.copy(eigvec0), np.copy(Vinv0)
Mvol = np.copy(Mvol0) 
Miatm = np.copy(Miatm0)
mc = np.copy(mc0)
rcore = np.copy(rcore0)
simulatedyears = 0
Micore = np.copy(Micore0)
pebblerate = np.copy(pebblerate0)

# %% Transport loop
timearray = np.zeros(number_of_timesteps//savingspacing+2) # create array to match time step with years simulated
Micoretimearray = np.zeros((len(volatiles), number_of_timesteps//savingspacing+2)) # create array to fill with volatile mass in the core
Miatmtimearray = np.zeros((len(volatiles), number_of_timesteps//savingspacing+2)) # create array to fill with volatile mass in the atmosphere

Mcoretimearray = np.zeros(number_of_timesteps//savingspacing+2) # create array to fill with total core mass
Mmagmatimearray = np.zeros(number_of_timesteps//savingspacing+2) # create array to fill with total magma ocean mass
protomasstimearray = np.zeros(number_of_timesteps//savingspacing+2) # create array to fill with total mass
pebbleratetimearray = np.zeros(number_of_timesteps//savingspacing+2) # create array to fill with pebble mass accretion rate

surfcondtimearray = np.zeros((3, number_of_timesteps//savingspacing+2)) # create array to fill with surface conditions
cmbcondtimearray = np.zeros((3, number_of_timesteps//savingspacing+2)) # create array to fill with conditions at the core mantle boundary 

savingindex = 0 # index for where in the timearrays to save element

timeplotarray = np.zeros(number_of_timesteps//plotstepspacing+2) # create an array to store the simulated time in years at each point a plot is made
Mvolplotarray = np.zeros((number_of_timesteps//plotstepspacing+2, len(volatiles), number_of_layers)) # create array to store each volatile mass in each layer
Pvolplotarray = np.zeros((number_of_timesteps//plotstepspacing+2, len(volatiles), number_of_layers)) # create array to store each volatile partition coefficient in each layer
structureplotarray = np.zeros((number_of_timesteps//plotstepspacing+2, 5, number_of_layers))
plotnumber = 0 # index for where in the plotarrays to save element

loopstart = time.time()
for timestep in range(number_of_timesteps):
    dt = stdmax**2/(2*D) # update dt as D changes
    simulatedyears += dt # add dt to the simulated time 

    # volatile motion inside magma ocean
    Mvol, Micore, corediff = transMvol(Mvol, Micore, eigval, eigvec, Vinv, dt, vmet) # transport using linalg equations for the change in mass in each layer
    Mtot = Msil+Mmet+np.sum(Mvol, axis=0)
    Mmagma = np.sum(Mtot) # sum of all magma ocean layers
    mc = mc+np.sum(corediff) # add volatile mass to core mass
    
    # magma ocean - atmosphere boundary interaction
    eqstate = Miatmeqfunc(Miatm, rsurf, amui, alphai, Msil[-1], Mvol[:,-1], fO2surf, protomass, pebblerate)
    Miatm = eqstate[0]
    Mvol[:,-1] = eqstate[1] # update magma ocean top layer's volatile content
    Matm = np.sum(eqstate[0])
    
    # magma ocean restructuring
    if timestep % restructuringspacing == 0 or timestep == number_of_timesteps-1:
        # Run without sedimentation for the first nosedifrac fraction of the total time to be simulated.
        if simulatedyears>settime*nosedifrac: 
            vmet = 1e-3*secondsperyear  # 1 m/s in km yr-1, falling speed of metal blobs
        
        if growthbool: # if we want to grow the planet
            mc += restructuringspacing*pebblerate*pebblemetalfrac*dt # grow core by directly adding core mass
            rcore = (3*mc/(4*pi*coremetaldensity))**(1/3) # calculate new core radius
            Msil[-1] += restructuringspacing*pebblerate*(1-pebblemetalfrac)*dt # add silicate mass to top layer (keep top layer density the same)
            rk = newlayerboundaries(Msil, density, rcore, number_of_boundaries) # See where all the boundaries should be such that the desired layer densities create the known layer masses. number_of_boundaries instead! to keep rk length.
            # total mass of planet
            protomass = mc+Mmagma+np.sum(Miatm)+np.sum(Mvol)
            pebblerate = protomass/tau
        rk, psurf, Tsurf, pressure, temperature, density, D, compressioncalctime = restructuring(rk, Msil, Matm, mc, pebblerate, number_of_structure_boundaries, 1, Miatm=Miatm, species='mix', plotting=False)
        dr = rk[1]-rk[0] # same layerwidth for all layers
        rsurf = rk[-1]  # new surface is the compressed surface
        layercenters = centeraveraging(rk)
        layervolumes = volumefunc(rk) # calculate volume of each magma ocean layer 
        Mmet = np.where(vmet!=0, pebblerate*pebblemetalfrac*dr/vmet, 0) # metal inside each layer, LET PEBBLERATE CHANGE?
        Msil = density*layervolumes # compressed density*volume is total mass of metal + silicates
        Msilmet = Msil+Mmet # subtract mass of metal to get mass of silicates in each magma ocean layer 
        Mtot = Msil+Mmet+np.sum(Mvol, axis=0) # total mass
        
        PH2O, CH2Omet = iterativePH2O(pressure, Mvol[0], Msil, Mmet, CH2Ometguess=CH2Omet)
        PC = PCfunc(pressure, temperature, 0)
        PN = PNfunc(pressure, temperature, XFeO)
        Pvol = np.array([PH2O, PC, PN]) # array of all the volatile species' partition coefficients (at all magma ocean layers)        
        
        # matrix of coefficients Ai updates every restructuring time step. Where Ai is defined from dMi/dt = Ai Mi
        eigval, eigvec, Vinv = eigandVinv(rk, Msil, Mmet, vmet, D, Pvol)

    # Storing time step variables 
    if timestep % savingspacing == 0 or timestep == number_of_timesteps-1:
        timearray[savingindex] = simulatedyears
        Micoretimearray[:, savingindex] = Micore # store volatile mass in core for each time step 
        Miatmtimearray[:, savingindex] = Miatm # store volatile mass in atmosphere for each time step
        
        Mcoretimearray[savingindex] = mc # store total mass of core
        Mmagmatimearray[savingindex] = Mmagma # store total mass of the magma ocean
        protomasstimearray[savingindex] = protomass # store mass of entire planet for each time step
        
        surfcondtimearray[:, savingindex] = np.array([rk[-1], psurf, Tsurf])
        cmbcondtimearray[:, savingindex] = np.array([rk[0], pressure[0], temperature[0]])
        
        savingindex+=1 # increment saving index

    if timestep % plotstepspacing == 0 or timestep == number_of_timesteps-1:
        volfrac = Mvol/Mtot # only calculate volatile mass fraction if it is plotted
        
        timeplotarray[plotnumber] = simulatedyears # at what times are the plots made
        structureplotarray[plotnumber] = np.array([layercenters, pressure, temperature, density, Mtot])
        Mvolplotarray[plotnumber] = Mvol
        Pvolplotarray[plotnumber] = Pvol
        
        # Plot distribution of volatiles
        distfig = plt.figure(figsize=(7,7))
        plt.title('Time step '+str(timestep+1)+', '+"{:.1f}".format(simulatedyears)+' years. '+r"$M_{\mathrm{atm}} = $"+"{:.1f}".format(Matm/Matm0)+r"$\ M_{\mathrm{atm}, 0}$"+'\n Volatile core mass fraction: '+"{:.2E}".format(np.sum(Micore)/mc)+', change: '+"{:.2E}".format(np.sum(corediff)/mc)+'\n proto mass: '+"{:.2E}".format(protomass/earthmass)+' '+r"$M_{\mathrm{E}}$", y=1.10)
        for i in irange:
            plt.semilogy(layercenters, volfrac0[i], color=vcolors[i], label=volatiles[i])
            plt.semilogy(layercenters, volfrac[i], '.', color=vcolors[i])
            plt.title('Volatile distribution in magma ocean \n'+'Time step '+str(timestep+1)+', '+"{:.1f}".format(simulatedyears)+' years. '+r"$M_{\mathrm{atm}} = $"+"{:.1f}".format(Matm/Matm0)+r"$\ M_{\mathrm{atm}, 0}$"+'\n Volatile core mass fraction: '+"{:.2E}".format(np.sum(Micore)/mc))
            plt.xlabel('radius [km]')
            plt.ylabel('Volatile mass fraction')
            plt.axis((None, None, min(Xi)*1e-2, max(Xi)*2)) 
            plt.grid()

        formatstring = '{:0>'+str(int(np.log10(max_number_of_plots+2)+1))+'}'
        plotstring = formatstring.format(plotnumber)
        distfig.savefig('Plots/VideoPlots/distfig'+plotstring+'.png', bbox_inches='tight')

        # Plot partition function
        partfig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,4))
        partfig.suptitle('Metal droplets/silicate melt partition coefficient of volatile species')
        
        ax1.set_prop_cycle('color', vcolors)
        ax1.semilogy(layercenters, np.transpose(Pvol), label=volatiles)
        ax1.set_xlabel('radius [km]')
        ax1.set_ylabel('Partition coefficient')
        ax1.set_ylim([1e-1, 5e2])
        ax1.legend()
        ax1.grid(True)

        ax2.set_prop_cycle('color', vcolors)
        ax2.semilogy(pressure, np.transpose(Pvol), label=volatiles)
        ax2.set_xlabel('Pressure [GPa]')
        ax2.set_ylim([1e-1, 5e2])
        ax2.grid(True)
        
        partfig.savefig('Plots/StructurePlots/PartFunc'+plotstring+'.png', bbox_inches='tight')        
        
        print('Plot #'+str(plotnumber)+' plotted')
        plotnumber+=1 # index for values stored for each time the plots appear
        
    
    print('Time step '+str(timestep+1)+'/'+str(number_of_timesteps)+' completed. t = '+"{:.2f}".format(simulatedyears)+' years. '+
          "{:.2f}".format(100*(timestep+1)/number_of_timesteps)+'% of time steps done, '+
          "{:.2f}".format(100*(protomass/earthmass-protomass0/earthmass)/(stoppingmass-massscaling))+'% of mass accreted.')
    # Check time to see if we need to halt code before reaching time limit
    checktime = time.time()
    # Check speed of the simulation so far
    currentlooptimeelapsed = checktime-loopstart
    currentsimulationspeed = simulatedyears/currentlooptimeelapsed
    print('Simulation speed = '+str(currentsimulationspeed)+' yr/s. '+'Time step speed = '+str(timestep/currentlooptimeelapsed))

    if (checktime-loopstart)>4.8*86400: # 5 days is maximum cluster time
        break # stop prematurely so that the code has time to save variables before 5 day node limit is reached
    if protomass>stoppingmass*earthmass: # if we have reached our desired mass break the loop
        break # stop prematurely when we have reached the correct mass


loopend = time.time()
looptimeelapsed = loopend-loopstart
simulationspeed = simulatedyears/looptimeelapsed # 3.7 years/second over 1e5 timesteps with only internal magma ocean motion, no surface interaction or restructuring. Half-speed (1.8) with restructuring + surface interaction. Approaches 2 for bigger number of time steps (5e5). 2 after subplots became more advanced
timestepspersecond = timestep/looptimeelapsed # timesteps per second
print('Time elapsed in looping: '+str(looptimeelapsed)+'. Simulation speed: '+str(simulationspeed)+'. Time steps per second: '+str(timestepspersecond))

#-----------------------------Saving the variables-----------------------------
# In case program was cut short, remove all elements that didn't get a chance to be filled. 
np.savetxt('time.out', timearray[:savingindex], delimiter=',') # 1d file that stores time in years at each time step (0D variables are recorded each time step)
np.savetxt('Surf.out', surfcondtimearray[:, :savingindex], delimiter=',') # 2d file. (rsurf, psurf, Tsurf) order for each saving time step
np.savetxt('CMB.out', cmbcondtimearray[:, :savingindex], delimiter=',') # 2d file. (rcmb, pcmb, Tcmb) order for each saving time step
np.savetxt('Misurf.out', Miatmtimearray[:, :savingindex], delimiter=',') # 2d file. Miatm for each saving time step
np.savetxt('Micore.out', Micoretimearray[:, :savingindex], delimiter=',') # 2d file. Micore for each saving time step
np.savetxt('Mcore.out', Mcoretimearray[:savingindex], delimiter=',') # 1d file. Mcore for each saving time step
np.savetxt('Mmagma.out', Mmagmatimearray[:savingindex], delimiter=',') # 1d file. mass of magma ocean for each saving time step
np.savetxt('protomass.out', protomasstimearray[:savingindex], delimiter=',') # 1d file. protomass for each saving time step

np.savetxt('timeplot.out', timeplotarray[:plotnumber], delimiter=',') # file that stores time in years at each time step where plots are produced or 1D variables are recorded
np.savetxt('rk.out', structureplotarray[:,0][:plotnumber], delimiter=',') # rows are layer boundaries for a given t (corresponding to timeplot.out)
np.savetxt('pressure.out', structureplotarray[:,1][:plotnumber], delimiter=',') # rows are pressure profile for a given t (corresponding to timeplot.out)
np.savetxt('temperature.out', structureplotarray[:,2][:plotnumber], delimiter=',') # -||- but temperature profile
np.savetxt('density.out', structureplotarray[:,3][:plotnumber], delimiter=',') # -||- but density profile
np.savetxt('Mtot.out', structureplotarray[:,4][:plotnumber], delimiter=',') # -||- but total mass in each layer
for i in irange: # save a txt-file for each volatiles' mass distribution in the magma ocean
    np.savetxt('M'+volatiles[i]+'.out', Mvolplotarray[:plotnumber,i], delimiter=',') # M of the specified volatile in each layer for each plotting time step
    np.savetxt('P'+volatiles[i]+'.out', Pvolplotarray[:plotnumber,i], delimiter=',') # partition coefficient of the specified volatile in each layer for each plotting time step
