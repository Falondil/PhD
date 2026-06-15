"""
Solving for Fe3+, Fe2+ oxidation equilibrium
assuming Fe + 1/2 O2 <-> FeO
and FeO + 1/4 O2 <-> FeO1.5 
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import fsolve
from scipy.integrate import quad

# CHOICES
Vtestplot = False # whether or not to plot the molar volumes of reaction components
IWpaper = ['Komabayashi', 'Hirschmann'][0] # paper source for KIW at DeltaIW = 0. 

# thermodynamic constants
R = 8.314 # J mol-1 K-1 

# magma ocean conditions
pressure = 50 # GPA, PLACEHOLDER
temperature = 3500 # K, PLACEHOLDER

# initial Fe and O configuration
xFeO = 0.056 # McDonough and Sun (1995). 5.6mol% FeO currently --> we assume 5.6mol% O^T, corresponding to 8.0wt% FeO. HOW IS IT 0.056???
eps = 1e-4 # make sure no division by 0 
nFe_0 = eps # number of Fe molecules
nFeO_0 = 1-2*eps # number of FeO molecules in units of the number of moles of FeO in Earth's mantle
nFeO15_0 = eps # number of FeO1.5 molecules

# nFetot0 = nFe_0 + nFeO_0 + nFeO15_0 # number of atoms of Fe in any component
# nOtot0 = nFeO_0 + 3/2*nFeO15_0 + 2*nO2_0 # number of atoms of O in any component

Xir0 = 0 # initial extent of 3 FeO(melt) <-> 2 FeO1.5(melt) + Fe(alloy) reaction

# magma ocean composition, McDonough&Sun1995, Table 4, Pyrolite model 1. This is what Hirschmann2022 uses for Table 1, first Earth row
Fecomponents = ['Fe', 'FeO', 'FeO1.5']
amu_comp = [55.8450, 71.844, 159.687/2] # g/moles of Fe

meltmolecules = np.array(['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'MnO', 'FeO', 'NiO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5'])
meltmassfrac = 1e-2*np.array([45.0, 0.201, 4.45, 0.384, 0.135, 8.05, 0.25, 37.8, 3.55, 0.36, 0.029, 0.021])
meltmassfrac /= sum(meltmassfrac) # normalize, total = 1.0023 otherwise

# MoleWeights.xlsx in Hirschmann2022 supplemental zip. Molar weights are also divided by number of cations in the molecule. Hence Al2O3 -> AlO1.5 is assigned ~50 amu instead of ~100 amu.
meltmoleweights = np.array([60.0735, 79.055, 50.971, 75.987, 71.839, 70.932, 40.299, 58.9634, 56.072, 30.9868, 47.095, 70.959]) 

frac = meltmassfrac/meltmoleweights
meltmolefrac = frac/np.sum(frac) # compute mole fraction of the melt components

# Molar Gibbs energies of O2. Chase 1998, NIST-JANAF. According to IWCalcHirschmmann2021.m the data is from "SGTE data for pure elements", Dinsdale 1991
def GChase(T):
    a, b, c, d, e, g = -13137.52, 25.32003, -33.627, -0.00119159, 525809.556, 0.00000001356111

    GO2 = a+b*T+c*T*np.log(T)+d*T**2+e/T
    # print('GO2 calcd. at 1 bar', GO2)
    return GO2

# Molar Gibbs energies of Fe and FeO liquids, Table 1. 
def GKomabayashi(T, component): # evaluates at 1 bar the Gibbs free energy per mole of a liquid Fe or liquid FeO. [J mol-1]
    if component=='Fe':
        a, b, c, d = -9007.3402, 290.29866,	-46, 0
    elif component=='FeO':
        a, b, c, d = -245310, 231.879, -46.12826, -0.0057402984
    else: 
        raise ValueError('Invalid component submitted as argument')
    return a+b*T+c*T*np.log(T)+d*T**2 # e and f = 0 for our components

# calculate equilibrium coefficient of forward liquid IW reaction, Fe + 1/2 O2 -> FeO
def iron_wustite_K_Komabayashi(P, T, Vtestplot=Vtestplot):
    # ----------Komabayashi 2014---------- 
    # V(T0, P) from inverting Eq. 3 (derived from compressibility via bulk modulus) and computing thermal expansion 
    def VKomabayashi(P, component):
        T0 = 298.15 # room-temperature
        if component == 'Fe': # liquid Fe, Table 2
            V0 = 6.88 # [cm3 mol-1], molar volume at 1 bar, 298 K
            K0 = 148 # [GPa], bulk modulus at 1 bar, 298 K
            Kprime = 5.8 # bulk modulus derivative w.r.t pressure at 1 bar, 298 K 
            alpha0 = 9e-5 # [K-1], thermal expansivity at 1 bar
            delta0 = 5.1 # Anderson-Grüneisen parameter at 1 bar
            Kcomponent = 0.56 # different K. component specific unitless parameter from Wood 1993
        elif component == 'FeO': # liquid FeO, Table 2
            V0 = 13.16 # [cm3 mol-1], molar volume at 1 bar, 298 K
            K0 = 128 # [GPa], bulk modulus at 1 bar, 298 K
            Kprime = 3.85 # bulk modulus derivative w.r.t pressure at 1 bar, 298 K 
            alpha0 = 4.7e-5 # [K-1], thermal expansivity at 1 bar
            delta0 = 4.5 # Anderson-Grüneisen parameter at 1 bar
            Kcomponent = 1.4 # different K. component specific unitless parameter from Wood 1993
        def P298rootfunc(V): # Eq. 3
            x = (V/V0)**(1/3)
            return 3*K0*x**(-2)*(1-x)*np.exp(3/2*(Kprime-1)*(1-x)) - P
        
        fac = 0.1 # unitless initial step size try
        Vguess = 0.95*V0 # starting guess. Expect slightly smaller after bulk compression + thermal expansion
        V_P_298 = fsolve(P298rootfunc, Vguess, xtol=1e-10, full_output=False, maxfev=400, factor=fac) # find roots (V(P) at 298 K) that solve Eq. 3. 
        
        #  thermal expansion
        alpha = alpha0*np.exp(-(delta0/Kcomponent*(1-(V_P_298/V0)**Kcomponent)))
        V_P_T = V_P_298*np.exp(alpha*(T-T0)) # Thermal expansion from definition, given in VinetVolumeFunc2.m inside Hirschmann2021 supplemental .zip

        VinSI = V_P_T*1e-6 # convert from cm3 to m3
        return VinSI

    # Checking how V(P) behaves
    if Vtestplot:
        Plin = np.linspace(0,100)
        V_P_Fe = np.array([VKomabayashi(Ptemp, component='Fe') for Ptemp in Plin])
        V_P_FeO = np.array([VKomabayashi(Ptemp, component='FeO') for Ptemp in Plin])

        plt.figure()
        plt.plot(Plin, V_P_Fe, color='r', label='Fe (liq)')
        plt.plot(Plin, V_P_FeO, color='b', label='FeO (liq)')
        plt.xlabel('Pressure [GPa]')
        plt.ylabel('Molar volume [m$^3$ mole$^{-1}$]')
        plt.title('Volumes of molecules involved in IW reaction')
        plt.legend()

    P0 = 0.0001 # [GPa], 1 bar is standard state
    integralVdP_Fe, integral_error_Fe = quad(VKomabayashi, P0, P, args=('Fe'))
    integralVdP_FeO, integral_error_FeO = quad(VKomabayashi, P0, P, args=('FeO'))
    integralDeltaVdP = 1e9*(integralVdP_FeO - integralVdP_Fe) # convert from GPa m^3 mole-1 to Pa m^3 mole-1 = J mole-1

    GFeliq = GKomabayashi(T, component='Fe')
    GFeOliq = GKomabayashi(T, component='FeO')
    GO2 = GChase(T)
    # print('GFe at 1 bar', GFeliq)
    # print('GFeO at 1 bar', GFeOliq)
    # print('GO2 at 1 bar', GO2)
    DeltaGIW1bar = GFeOliq - GFeliq - 1/2*GO2

    DeltaGIW = DeltaGIW1bar + integralDeltaVdP
    KIW = np.exp(-DeltaGIW/(R*T))

    # print('DeltaGIW1bar', DeltaGIW1bar)
    # print('Integral DeltaV_IW dP', integralDeltaVdP)
    # print('DeltaGIW', DeltaGIW)
    # print('KIW', KIW)

    return KIW

def iron_wustite_fO2(P, T): # Hirschmann2021 parametrization of log10 fO2 at (P, T) and DeltaIW = 0 for SOLID Fe and FeO.
    def m(m0, m1, m2, m3, m4):
        return m0+m1*P+m2*P**2+m3*P**3+m4*P**(1/2)
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

    hcpdomain = P>(-18.64 + 0.04359*T - 5.069e-6*T**2) # check for what p, T the iron is hcp
    def fit(P1, P2, P3, P4): # fitting function using either (a,b,c,d) or (e,f,g,h)
        return P1+P2*T+P3*T*np.log(T)+P4/T
    log10fO2 = np.where(hcpdomain, fit(e,f,g,h), fit(a,b,c,d))
    return 10**log10fO2

def iron_wustite_K_Hirschmann(P, T): # log10 fO2 = DeltaIW - 2log10(KIW) --> KIW log10 fO2_IW = -2log10(KIW) 
    fO2_IW = iron_wustite_fO2(P,T) # Hirschmann2021 parametrization
    KIW = fO2_IW**(-1/2)
    return KIW

def iron_wustite_K(P, T, paperchoice=IWpaper):
    if paperchoice=='Hirschmann':
        return iron_wustite_K_Hirschmann(P,T)
    elif paperchoice=='Komabayashi':
        return iron_wustite_K_Komabayashi(P, T)
    else:
        raise ValueError('Invalid paper choice for KIW')

# Hirschmann2022 says G_FeO from JANAF known to be erroneous. Can recalibrate using GKomabayashi(T, component='FeO')?
def DeltaG0Deng(T): # Supplementary note 1 for parametrization of change to molar Gibbs free energy from the reaction
    a, b, c, d, e, f = -3.310e5, -190.379, 14.785, -1.649e-3, 9.348e6, 1.077e4
    return a+b*T+c*T*np.log(T)+d*T**2+e*T**(-1)+f*T**(1/2)

# calculate equilibrium coefficient of forward simplified Fe3+ reaction, FeO + 1/4 O2 -> FeO1.5
def Fe3plusreact_K(P, T, Vtestplot=Vtestplot): 
    # ----------Deng 2020---------- 
    def VDeng(P, component): # calculates V_Mg14Fe2Si16O49 or V_Mg14Fe2Si16O48 by inverting fourth order Birch-Murnaghan equation of state for P(V,T0) at T0 = 3000 K and adding thermal expansion term B_TH (Eq. 2, Deng 2020)
        V0conv = 602.21 # convert from [Å^3 molecule-1] to [J GPa-1 mole-1]
        # 1 Å^3 / molecule = 1e-30 m^3 * 6.0221e23 mole-1
        # 1 m^3 = 1 J/Pa --> 1 m^3 = 1e9 J/GPa
        if component == 'red':
            # compressibility parameters of the reduced molecule (Mg14Fe2Si16O48)
            V0 = V0conv*1180.114014 # [J/GPa/mole], molar volume at 1 bar, 3000 K (not sure on 1 bar or 0 bar, but whatever. Hirschmann2022 supplement assumes it is at 1 bar)
            K0 = 26.94713861 # [GPa], bulk modulus at 1 bar, 3000 K 
            Kprime = 2.802531871 # bulk modulus derivative w.r.t pressure at 1 bar, 3000 K
            Kbis = 0.012313472 # [GPa-1], bulk modulus double derivative w.r.t pressure at 1 bar, 3000 K
            # thermal expansion parameters
            a = 35.79397483 # [GPa/K]
            b = 71.10313668 # [GPa/K]
            c = 36.59545225 # [GPa/K]
        elif component == 'ox':
            # parameters of the oxidized molecule (Mg14Fe2Si16O49)
            V0 = V0conv*1204.763652
            K0 = 23.19530062
            Kprime = 3.216089358
            Kbis = 0.009340183
            # thermal expansion parameters
            a = 34.52616394 
            b = 68.64429623
            c = 35.27069116    
        else: 
            raise ValueError('Invalid component submitted as argument')
        
        P0 = 0.0001 # [GPa], 1 bar is standard state
        T0 = 3000 # [K], reference temp. for Deng 2020
        def P3000rootfunc(V): # calculates V of assumed melt molecules in Deng 2020. (Eq. 6 & 7)
            V_P_3000 = ((3*V0*P0)+2*(3*V0*(3*K0-5*P0)/2)*(1/2*((V0/V)**(2/3)-1)) +
            + 3*(V0*(9*K0*Kprime - 36*K0 + 35*P0)/2)*((1/2*((V0/V)**(2/3)-1))**2) +
            + 4*(3*V0*(9*(K0**2)*Kbis + 9*K0*(Kprime**2)- 63*K0*Kprime + 143*K0 - 105*P0)/8.0)*((1/2*((V0/V)**(2/3)-1))**3))*(-(-(V0**(2/3))/3/(V**(5/3)))) # fourth order Birch-Murnaghan eos from BM4VolumeFunc.m included in Hirschmann 2022 supplemental zip
            B_TH = (a - b*(V/V0) + c*(V/V0)**2)/1000 # thermal pressure coefficient
            return V_P_3000 + B_TH*(T-T0) - P # zero from Eq. 2, Deng 2020
        
        fac = 0.1 # unitless initial step size try
        Vguess = 0.95*V0 # starting guess. Expect slightly smaller after bulk compression + thermal expansion
        V_P_T = fsolve(P3000rootfunc, Vguess, xtol=1e-10, full_output=False, maxfev=400, factor=fac) # roots (V(P) at 3000 K) from fourth order Birch-Murnaghan equation of state

        return V_P_T # [J/GPa/mole]
    
    P0 = 0.0001 # [GPa], 1 bar is standard state

    # Checking how V(P) behaves
    if Vtestplot:
        Plin = np.linspace(P0,100)
        V0conv = 602.21 # convert from [Å^3 molecule-1] to [J GPa-1 mole-1]
        V_red = 1/V0conv*np.array([VDeng(Ptemp, component='red') for Ptemp in Plin]) # [Å^3 molecule-1]
        V_ox = 1/V0conv*np.array([VDeng(Ptemp, component='ox') for Ptemp in Plin]) # [Å^3 molecule-1]

        plt.figure()
        plt.plot(Plin, V_red, color='r', label='reduced')
        plt.plot(Plin, V_ox, color='b', label='oxidized')
        plt.xlabel('Pressure [GPa]')
        plt.ylabel('Molar volume [Å$^{3}$ molecule$^{-1}$] at T = 3000 K')
        plt.legend()
        plt.xlim(left=P0)
        plt.ylim(top=1400)
        plt.title('Volumes of molecules involved in Fe3+ reaction')
        
        # # Checking how Delta V dP behaves
        plt.figure()
        delta_V_cm_mole = 1/2*(V_ox-V_red)*V0conv*1e6*1e-9 # [Å^3 molecule-1] to [J GPa-1 mole-1] to [cm^3 mole-1]
        plt.plot(Plin, delta_V_cm_mole, color='b')
        plt.xlabel('Pressure [GPa]')
        plt.ylabel('Molar volume [cm$^{3}$ mole$^{-1}$] at T = 3000 K')
        plt.legend()
        plt.xlim(left=P0)
        plt.ylim(top=12)
        plt.title('Molar volume change from Fe3+ reaction')
    
    integralVdP_red, integral_error_red = quad(VDeng, P0, P, args=('red')) # [J mole-1], VDeng returns volumes in units [J GPa-1 mole-1] and then quad integrates, multiplying by units of P = [GPa]
    integralVdP_ox, integral_error_ox = quad(VDeng, P0, P, args=('ox')) # [J mole-1]
    integralDeltaVdP = 1/2*(integralVdP_ox - integralVdP_red) # Eq. 6, Deng 2020. Approximate V_FeO1.5 - V_FeO = 1/2 (V_Mg14Fe2Si16O49 - V_Mg14Fe2Si16O48)
    
    Delta_GFe3plus0 = DeltaG0Deng(T) # molar Gibbs free energy change at P0, T
    Delta_GFe3plus = Delta_GFe3plus0 + integralDeltaVdP # molar Gibbs free energy change at P, T
    KFe3plus = np.exp(-Delta_GFe3plus/(R*T)) # equilibrium coefficient of the reaction

    # print('Delta_GFe3plus0', Delta_GFe3plus0)
    # print('Integral DeltaV_Fe3+ dP', integralDeltaVdP)
    # print('Delta_GFe3plus', Delta_GFe3plus)
    # print('KFe3plus', KFe3plus)

    # BE SUPER CAREFUL THAT THE ACTUAL RESULT DOES
    # NOT BECOME NEGATIVE, K_COMB = K_3+^2/K_IW.
    # K_3+ < 0 IS THEREFORE HARDER TO SPOT AS A 
    # CLEAR ERROR THAN K_IW < 0

    return KFe3plus

# Compute the equilibrium coefficients
KIW = iron_wustite_K(pressure, temperature)
KFMQ = Fe3plusreact_K(pressure, temperature)

# iterates line search for Delta_r g -> 0 (i.e., chemical equilibrium)
def equilibrium_finder(nFe=nFe_0, nFeO=nFeO_0, nFeO15=nFeO15_0, KIW=KIW, KFMQ=KFMQ, Xir0 = Xir0, OperFe=1, xO=xFeO, max_iter=1000, tol=1e-8):
    Xir = Xir0
    Xir_list = [Xir]
    dXir_old = 0
    alpha0 = 1 # initial guess for alpha for first iteration

    for _ in range(max_iter):
        nFe, nFeO, nFeO15, dXir, distance_to_equilibrium, alpha0 = line_search_reaction_extent(nFe, nFeO, nFeO15, KIW, KFMQ, OperFe=OperFe, xO=xO, alpha0=alpha0)

        Xir += dXir
        Xir_list.append(Xir)

        if distance_to_equilibrium < tol:
            break

        if dXir_old*dXir<0:
            alpha0 *= 0.5
        
        dXir_old = dXir
    
        if _ == max_iter-1:
            print('Something is fishy, distance_to_equilibrium = ', distance_to_equilibrium, ' at O/Fe = ', OperFe, 'nFe =', nFe)

    return nFe, nFeO, nFeO15, Xir_list


def line_search_reaction_extent(nFe, nFeO, nFeO15, KIW, KFMQ, OperFe=1, xO=xFeO, alpha0=1, activitymethod=2):
    
    # compute activities
    aFe, aFeO, aFeO15 = n_to_activity(nFe, nFeO, nFeO15, OperFe=OperFe, xO=xO, method=activitymethod)

    Kr = (KFMQ**2)/KIW # 3 FeO <-> 2 FeO1.5 + Fe is 2*"FMQ" - IW
    log_Kr = np.log10(Kr)
    log_Qr = np.log10(aFe)+2*np.log10(aFeO15)-3*np.log10(aFeO)

    Delta_gr = log_Qr-log_Kr # unitless Gibbs free energy change of the reaction
    distance_to_equilibrium_0 = Delta_gr**2

    # line search
    alpha = alpha0 
    shrink = 1/2
    while True:
        dXir = -alpha*Delta_gr # how much forward reaction

        nFe_1, nFeO_1, nFeO15_1 = extent_to_n(nFe, nFeO, nFeO15, dXir)
        if min(nFe_1, nFeO_1, nFeO15_1) < 0: # no negative n allowed
            alpha *= shrink
            continue
        
        aFe_1, aFeO_1, aFeO15_1 = n_to_activity(nFe_1, nFeO_1, nFeO15_1, OperFe=OperFe, xO=xO, method=activitymethod)


        log_Qr_1 = np.log10(aFe_1)+2*np.log10(aFeO15_1)-3*np.log10(aFeO_1)
        Delta_gr_1 = log_Qr_1 - log_Kr # unitless Gibbs free energy of reaction

        distance_to_equilibrium_1 = Delta_gr_1**2

        if distance_to_equilibrium_1 < distance_to_equilibrium_0:
            break # new component config. is closer to equilibrium
        else:
            alpha *= shrink # new component config. is further from equilibrium, overstepping

        if alpha < 1e-14: # don't go too small
            # print('Something is fishy')
            break
    return nFe_1, nFeO_1, nFeO15_1, dXir, distance_to_equilibrium_1, alpha


def extent_to_n(nFe_0, nFeO_0, nFeO15_0, dXir):
    nFe = nFe_0 + dXir
    nFeO = nFeO_0 - 3*dXir
    nFeO15 = nFeO15_0 + 2*dXir
    return nFe, nFeO, nFeO15
    

def n_to_activity(nFe, nFeO, nFeO15, OperFe=1, xO=xFeO, method=2):
    # Alternative 0, PLACEHOLDER. Simply assume molefraction = activity 
    if method == 0:
        ntot = nFe+nFeO+nFeO15
        if ntot <= 0:
            raise ValueError('Negative total moles of condensed components')
        return np.array([nFe, nFeO, nFeO15])/ntot
    
    elif method == 1 or method == 2:
        ntot = nFe+nFeO+nFeO15 # Fe^T normalization
        if ntot <= 0:
            raise ValueError('Negative total moles of condensed components')
        xFeT = xO/OperFe # actual xFe^T in magma ocean mole fraction. xFeT = xO/(O/FeT) 
        Ntot = ntot/xFeT # total number of molecules in magma ocean in units of nFeT
        nonFemet = Ntot/1e16 # PLACEHOLDER, arbitrarily tiny mol% of magma ocean moles are non-Fe metal (like Ni or whatever). This is needed to ensure a_Fe -> 0 is possible s.t. Qr can take all positive real number values. I.e. equilibrium exists, even for vanishingly low nFe. 
        xFeinmet = nFe/(nFe+nonFemet) 
        xFeOinsil = nFeO/Ntot # molefraction FeO in silicate
        xFeO15insil = nFeO15/Ntot # molefraction FeO1.5 in silicate
        
        # Alternative 1, PLACEHOLDER: Still simply assume molefraction its phase = activity
        if method == 1:
            a_Fe, a_FeO, a_FeO15 = xFeinmet, xFeOinsil, xFeO15insil

        # Alternative 2: Use activity coefficients (a_i = gamma_i x_i)
        elif method == 2:
            gamma_Fe = 1 # PLACEHOLDER, consistent with first Earth row of Table 1 in Hirschmann2022 using Fe in alloy from McDonough2003
            # gamma_Fe = 0.86 # PLACEHOLDER, (McDonough&Sun1995) 86% of Fe in core, 14% in silicate mantle. We assume 14% of core-destined metal is other elements

            TJaya = 1682 # K, temperature of Jayasuriya2004 experiments
            # (Fe3+/FeT at fO2 = 0) = 1.861 from Jayasuriya2004. Just below Table 4.
            activityratio = np.exp(-DeltaG0Deng(TJaya)/(R*TJaya)-1.861) # gamma_FeO1.5/gamma_FeO at fO2 = 0. Uses recent DeltaG0 for the FeO + 1/4 O2 -> FeO1.5 reaction from Deng2020
            
            gamma_FeO = 1.55 # CHOICE in Hirschmann 2022
            gamma_FeO15_Jaya = activityratio*gamma_FeO
            
            a_Fe, a_FeO, a_FeO15 = gamma_Fe*xFeinmet, gamma_FeO*xFeOinsil, gamma_FeO15_Jaya*xFeO15insil 

        return a_Fe, a_FeO, a_FeO15
    
    else:
        raise ValueError('Invalid method chosen')
    


# %% Finding equilibrium

starttime = time.time()
res = equilibrium_finder()
endtime = time.time()
time_elapsed = endtime-starttime
print(time_elapsed, 'seconds to find equilibrium')

# How does reaction extents converge?
Xir_array = np.array(res[3])

plt.figure()
plt.title('Reaction extents')
plt.plot(Xir_array, color='r', label='$\\xi_{\mathrm{r}}$')
plt.legend()
plt.grid(alpha=0.4)

# # How does moles converge? 
n_list = extent_to_n(nFe_0, nFeO_0, nFeO15_0, Xir_array)
plt.figure()
plt.title(f'Equilibrium convergence at O/Fe = 1\n(P, T) = ({pressure:.0f} GPa, {temperature:.0f} K)')
plt.plot(n_list[0], label='Fe')
plt.plot(n_list[1], label='FeO')
plt.plot(n_list[2], label='FeO$_{1.5}$')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Fe$^{\mathrm{(component)}}$/Fe$^{\mathrm{T}}$')
plt.grid(alpha=0.4)

# %% Equilibrium finder for array of pressure and temperature

# Alternative 0. Assume linear (P, T) structure
# Plin = np.linspace(1, 100, nlayers)
# Tlin = np.linspace(3000, 4000, nlayers) 

# Alternative 1. Use internal (P, T) from end of some VolatileTransportX.py simulation
import pickle
with open('continuefile', 'rb') as f:
        simulatedyears, protomass0, pebblerate0, mc0, Msil, Mmet, rsurf, rk, rcore, psurf, Tsurf, Plin, Tlin, density, D, CH2Omet, Pvol, Mvol, Micore0, Miatm, Miatmextra0, fO2surf0, eigval0, eigvec0, Vinv0 = pickle.load(f)
nlayers = len(Plin)
nspecies = 3 # Fe, FeO, FeO1.5

starttime2 = time.time()
equilibrium_compositions = np.zeros((nlayers, nspecies))
Kstarttime = np.zeros(nlayers)
Kendtime = np.zeros(nlayers)
for j, (P_k, T_k) in enumerate(zip(Plin, Tlin)):
    Kstarttime[j] = time.time()
    KIW_k = iron_wustite_K(P_k, T_k)
    KFMQ_k = Fe3plusreact_K(P_k, T_k)
    Kendtime[j] = time.time()
    res_k = equilibrium_finder(KIW = KIW_k, KFMQ = KFMQ_k)
    # Xir_k_eq = res_k[3][-1] # reaction extent at equilibrium
    # n_list_k = extent_to_n(nFe_0, nFeO_0, nFeO15_0, Xir_k_eq)
    n_list_k = res_k[:3] # simpler to just take it
    equilibrium_compositions[j,:] = n_list_k
endtime2 = time.time()
Ktotaltime = sum([Ke-Ks for Ke, Ks in zip(Kendtime, Kstarttime)])

fig, ax = plt.subplots()
ax.set_title('Component speciation at O/Fe = 1\nas function of (P,T)')
for j in range(nspecies):
    ax.plot(equilibrium_compositions[:,j], Plin, label=['Fe', 'FeO', 'FeO$_{1.5}$'][j])
ax.legend()
ax.set_xlabel('Fe$^{\mathrm{(component)}}$/Fe$^{\mathrm{T}}$')
ax.set_ylabel('P [GPa]')
ax.grid(alpha=0.4)
ax.set_ylim(bottom=0, top=80)
ax.set_xlim(left=0)

secax = ax.secondary_yaxis(
    'right',
    functions=(
        lambda P: np.interp(P, Plin[::-1], Tlin[::-1]),  # P -> T
        lambda T: np.interp(T, Tlin[::-1], Plin[::-1])   # T -> P
    )
)
secax.set_ylabel('T [K]')

print('Time to compute equilibrium in all', nlayers, 'layers:', endtime2-starttime2, 's')
print('Time to compute equilibrium constants in all', nlayers, 'layers:', Ktotaltime, 's')

# %%% 
# Fe/O plot for a few (P, T)
linestylearray = [':','-.','--','-']
speciescolors = ['tab:blue', 'tab:orange', 'tab:green'] # Fe, FeO, FeO1.5 in that order
Ptestarray = np.array([0, 25, 50, 75]) 
Ttestarray = np.interp(Ptestarray, Plin[::-1], Tlin[::-1])

OperFe_zoom = False # CHOICE, to zoom in at OperFe \in [0.9, 1.1] or not

s = 20 # size, number of O/Fe values to plot
safety = 0.01 # small distance from theoretical min and max values 
if OperFe_zoom:
    OperFe_min = 0.9
    OperFe_max = 1.1
else:
    OperFe_min = 0+safety # O/Fe = 0 if all is Fe 
    OperFe_max = 1.5-safety # O/Fe = 1.5 if all is FeO1.5
OperFe_array = np.linspace(OperFe_min, OperFe_max, s) # O/Fe, ratio of number of atoms

# assume some amount of Fe and some amount of FeO1.5 to get desired O/Fe ratio
N = 1 # arbitrary unit amount
nFeO_0_array = eps+np.zeros(s) # arbitrarily = 0 at start
nFeO15_0_array = N*(OperFe_array-nFeO_0_array)/1.5
nFe_0_array = N-nFeO15_0_array-nFeO_0_array

# print('nFe_0_array', nFe_0_array)
# print('nFeO_0_array', nFeO_0_array)
# print('nFeO15_0_array', nFeO15_0_array)

# compute all equilibrium coefficients
KIW_test_array = np.zeros(len(Ptestarray))
KFMQ_test_array = np.zeros(len(Ptestarray))
for j, (Ptest, Ttest) in enumerate(zip(Ptestarray, Ttestarray)):
    KIW_test_array[j] = iron_wustite_K(Ptest, Ttest)
    KFMQ_test_array[j] = Fe3plusreact_K(Ptest, Ttest)

# find equilibria
equilibrium_compositions_2 = np.zeros((len(Ptestarray),  nspecies, s)) # matrix for storing molecular component fractions
equilibrium_activities_2 = np.zeros((len(Ptestarray), nspecies, s)) # matrix for storing activities
fO2_equilibrium_array_IW = np.zeros((len(Ptestarray), s)) # for each layer (P,T) and each O/Fe
fO2_equilibrium_array_FMQ = np.zeros((len(Ptestarray), s)) # for each layer (P,T) and each O/Fe
DeltaIW_array = np.zeros((len(Ptestarray), s)) # for each layer and assumed O/Fe, calculate DeltaIW

for j1 in range(len(Ptestarray)):
    for j3, (nFe_0_test, nFeO_0_test, nFeO15_0_test) in enumerate(zip(nFe_0_array, nFeO_0_array, nFeO15_0_array)):
        res = equilibrium_finder(
                nFe=nFe_0_test, nFeO=nFeO_0_test, nFeO15=nFeO15_0_test,
                KIW = KIW_test_array[j1], KFMQ = KFMQ_test_array[j1], OperFe=OperFe_array[j3], xO=xFeO
            )
        Xir_eq = res[3][-1] # reaction extent at equilibrium
        n_list = res[:3] # easier, # n_list = extent_to_n(nFe_0=nFe_0_test, nFeO_0=nFeO_0_test, nFeO15_0=nFeO15_0_test, dXir=Xir_eq)
        equilibrium_compositions_2[j1, :, j3] = n_list 
        
        # compute fO2 given an equilibrium composition, KIW and K'FMQ'
        # fO2 = (a_FeO1.5/(a_FeO*K_FMQ))^4 or fO2 = (a_FeO/(a_Fe*K_IW))^2
        a_Fe, a_FeO, a_FeO15 = n_to_activity(n_list[0], n_list[1], n_list[2], OperFe=OperFe_array[j3], xO=xFeO)
        equilibrium_activities_2[j1, :, j3] = np.array([a_Fe, a_FeO, a_FeO15]) 
        
        fO2_equilibrium_array_IW[j1, j3] = (a_FeO/(a_Fe*KIW_test_array[j1]))**2
        fO2_equilibrium_array_FMQ[j1, j3] = (a_FeO15/(a_FeO*KFMQ_test_array[j1]))**4
        DeltaIW_array[j1, j3] = 2*np.log10(a_FeO/a_Fe) # Shouldn't work? a_Fe becomes undefined at high O/Fe.
        # DeltaIW_array[j1, j3] = np.log10(fO2_equilibrium_array_FMQ[j1, j3])+2*np.log10(KIW_test_array[j1]) # equivalent statement, works if aFe becomes undefined

print(equilibrium_compositions_2.shape)
print(fO2_equilibrium_array_IW.shape)
print(fO2_equilibrium_array_FMQ.shape)

# ---------- Component fraction plots ----------
# mole fraction of each component
plt.figure()
for j1 in range(len(Ptestarray)):
    for j2 in range(nspecies):
        plt.plot(OperFe_array, equilibrium_compositions_2[j1, j2, :], color=speciescolors[j2], linestyle=linestylearray[j1])

PT_handles = []
for j1 in range(len(Ptestarray)):
    PT_handle, = plt.plot([],[],color='k',linestyle=linestylearray[j1], label=(f'P={Ptestarray[j1]:.0f} GPa\nT={Ttestarray[j1]:.0f} K'))
    PT_handles.append(PT_handle)
PT_legend = plt.legend(handles=PT_handles, title='Conditions', bbox_to_anchor=(1.0, 0.70), loc='upper left')

species_handles = []
for j2 in range(nspecies):
    species_handle, = plt.plot([],[],color=speciescolors[j2],label=['Fe/Fe$^{\mathrm{T}}$','FeO/Fe$^{\mathrm{T}}$','FeO1.5/Fe$^{\mathrm{T}}$'][j2])
    species_handles.append(species_handle)
species_legend = plt.legend(handles=species_handles, title='Species', bbox_to_anchor=(1, 1), loc='upper left')
plt.gca().add_artist(PT_legend)

plt.xlabel('O/Fe')
plt.ylabel('Component mole fraction')
plt.title('Mole speciation varying total O/Fe\nx$_{\mathrm{O}}$='+f'{xFeO*100:.1f}'+'mol%')

plt.xlim(left=OperFe_min-safety, right=OperFe_max+safety)
plt.ylim(bottom=0, top=1)

plt.axvline(1, linestyle='-', alpha=0.2, color='k')
plt.axis=('equal')

# mass fraction of each component
component_weight_fraction = np.zeros((len(Ptestarray), nspecies, s))
for j1 in range(len(Ptestarray)):
    for j2 in range(nspecies):
        for j3 in range(s):
            frac = equilibrium_compositions_2[j1,:,j3]/amu_comp # component mole fraction / amu_component
            component_weight_fraction[j1, j2, j3] = frac[j2]/sum(frac) # weight fraction of each Fe-bearing component normalized to Fe^T

plt.figure()
for j1 in range(len(Ptestarray)):
    for j2 in range(nspecies):
        plt.plot(OperFe_array, component_weight_fraction[j1, j2, :], color=speciescolors[j2], linestyle=linestylearray[j1])

PT_legend = plt.legend(handles=PT_handles, title='Conditions', bbox_to_anchor=(1.0, 0.70), loc='upper left')

species_legend = plt.legend(handles=species_handles, title='Species', bbox_to_anchor=(1, 1), loc='upper left')
plt.gca().add_artist(PT_legend)

plt.xlabel('O/Fe')
plt.ylabel('Component mass fraction')
plt.title('Mass speciation varying total O/Fe\nx$_{\mathrm{O}}$='+f'{xFeO*100:.1f}'+'mol%')

plt.xlim(left=OperFe_min-safety, right=OperFe_max+safety)
plt.ylim(bottom=0, top=1)

plt.axvline(1, linestyle='-', alpha=0.2, color='k')
plt.axis=('equal')


# ---------- activities plot ----------
plt.figure()
for j1 in range(len(Ptestarray)):
    for j2 in range(nspecies):
        plt.plot(OperFe_array, equilibrium_activities_2[j1, j2, :], color=speciescolors[j2], linestyle=linestylearray[j1])

PT_legend = plt.legend(handles=PT_handles, title='Conditions', bbox_to_anchor=(1.0, 0.70), loc='upper left')

species_legend = plt.legend(handles=species_handles, title='Species', bbox_to_anchor=(1, 1), loc='upper left')
plt.gca().add_artist(PT_legend)

plt.xlabel('O/Fe')
plt.ylabel('Component activities')
plt.title('Activities varying total O/Fe\nx$_{\mathrm{O}}$='+f'{xFeO*100:.1f}'+'mol%')

plt.xlim(left=OperFe_min-safety, right=OperFe_max+safety)
plt.ylim(bottom=0, top=1)

plt.axvline(1, linestyle='-', alpha=0.2, color='k')
plt.axis=('equal')


# ---------- fO2 plot ----------
plt.figure()
for j1 in range(len(Ptestarray)):
    plt.semilogy(OperFe_array, fO2_equilibrium_array_IW[j1], color='k', linestyle=linestylearray[j1])
    plt.semilogy(OperFe_array, fO2_equilibrium_array_FMQ[j1], color='gray', linestyle=linestylearray[j1])

PT_legend = plt.legend(handles=PT_handles, title='Conditions', bbox_to_anchor=(1.0, 0.70))

reaction_handles = []
for j2 in range(2):
    reaction_handle, = plt.plot([],[],color=['k', 'grey'][j2],label=['IW-derived fO2','FMQ-derived fO2'][j2])
    reaction_handles.append(reaction_handle)
reaction_legend = plt.legend(handles=reaction_handles, title='Reaction', bbox_to_anchor=(1, 1))
plt.gca().add_artist(PT_legend)

plt.xlabel('O/Fe')
plt.ylabel('fO2 [bar]')
plt.title('Oxygen fugacity\nx$_{\mathrm{O}}$='+f'{xFeO*100:.1f}'+'mol%')

plt.xlim(left=OperFe_min-safety, right=OperFe_max+safety)

plt.axvline(1, linestyle='-', alpha=0.2, color='k')
plt.axis=('equal')


# ---------- DeltaIW plot ----------
plt.figure()
for j1 in range(len(Ptestarray)):
    plt.plot(OperFe_array, DeltaIW_array[j1], color='k', linestyle=linestylearray[j1])
    
PT_legend = plt.legend(handles=PT_handles, title='Conditions', bbox_to_anchor=(1.0, 1.0))

plt.xlabel('O/Fe')
plt.ylabel('$\Delta$IW')
plt.title('Oxygen fugacity distance to the IW buffer')

plt.xlim(left=OperFe_min-safety, right=OperFe_max+safety)

plt.axhline(-2, linestyle='-', alpha=0.2, color='k')
plt.axvline(1, linestyle='-', alpha=0.2, color='k')
plt.axis=('equal')

# ---------- Compute and plot the sedimentation time scale ----------
# sedimentation rate = dM_{Fe^T}^{magma ocean} / dt = - M_{Fe^T}^{magma ocean} * M^{bottom layer}/M^{magma ocean} * v_met/(layer width) * Fe(met)/FeT 
# Fe_weight_fraction = component_weight_fraction[:,0,:] # obsolete
Fe_mole_fraction = equilibrium_compositions_2[:, 0, :]/np.sum(equilibrium_compositions_2, axis=1) # should be normalized already, but summation just to be sure

secondsperyear = 31556926  # 31 556 926 seconds per year
vmet = 1*1e-3*secondsperyear # [km/yr]. Converted from 1 m/s
Deltar0 = rk[1]-rk[0] # [km], layer width of lowest magma ocean layer
fallingrate = vmet/Deltar0 # time to fall through one layer of the magma ocean (all layers have equal width)

# this also equals the O/Fe timescale!
sedimentationtimescale = (Fe_mole_fraction*Msil[0]/sum(Msil)*fallingrate)**(-1) # [yr], sedimentation time scale aka e-folding time scale

plt.figure()
for j1 in range(len(Ptestarray)):
    plt.semilogy(OperFe_array, sedimentationtimescale[j1], linestyle=linestylearray[j1], color='k')

PT_legend = plt.legend(handles=PT_handles, title='Conditions', bbox_to_anchor=(1.0, 0.70), loc='upper left')

plt.xlabel('O/Fe')
plt.ylabel('$-M_{\mathrm{Fe}^\mathrm{T}} \ dM_{\mathrm{Fe}^\mathrm{T}}$$^{-1} \ dt$  [yr]')
plt.title('Sedimentation time scale varying total O/Fe')

plt.xlim(left=OperFe_min-safety, right=OperFe_max+safety)

plt.axvline(1, linestyle='-', alpha=0.2, color='k')
plt.axis=('equal')

# %%% Evolution of O/Fe
# d(O/Fe)/dt = O/Fe * k_exponent * Fe(met)/FeT where Fe(met)/FeT is a function of O/Fe, P, and T. 
# (O/Fe)_1 = (O/Fe)_0 * exp(k_exponent * Fe(met)/FeT * \Delta t)
# to find (O/Fe)_1, evaluate Fe(met)/FeT using (O/Fe)_0

# extract rk for different planet sizes
rkfromfile = np.loadtxt('rk.out', delimiter=',')
pressurefromfile = np.loadtxt('pressure.out', delimiter=',')

rCMB_evo = rkfromfile[:,0]
PCMB_evo = pressurefromfile[:,0]

# computable before entering loop
Mbottomratio = np.interp(Ptestarray, Plin[::-1], Msil/sum(Msil)) # INCORRECT. Am dividing bottom layer with all layers, even if middle layer is assumed to be CMB. I.e. the ones below shouldn't count. mass of bottom layer / mass of magma ocean as function of what we assume bottom layer mass to be
# k_exponent = Mbottomratio*fallingrate # Exponent k in equation vmet/layer width * Mbottomratio
# layercenters = (rk[1:]+rk[:-1])/2 # center of each layer
rcore_test = np.interp(Ptestarray, PCMB_evo, rCMB_evo) # STILL NOT CORRECT. rcore_test should check at what rc that PCMB actually equalled Ptestarray 
k_exponent = 3*rcore**3/(rsurf**3-rcore**3)*vmet/rcore_test # more accurate, non-discretized expression

n_FeO_guess = 0+eps # initially guess n_FeO = 0 for every time step

# time loop choices
timestepfactor = 0.001 # pre-factor for time step size
max_time = 1e3 # [yr], arbitrary
max_timestep = max_time/1e2 # [yr], how big time steps do we allow.

# initializing matrices w.r.t time and (P, T) conditions
totaltimesteps_array = [0 for P in Ptestarray]
t_matrix = [0 for P in Ptestarray]
OperFe_matrix = [0 for P in Ptestarray]
DeltaIW_matrix = [0 for P in Ptestarray]
nFe_matrix_CMB = [0 for P in Ptestarray]
nFeO_matrix_CMB = [0 for P in Ptestarray]
nFeO15_matrix_CMB = [0 for P in Ptestarray]
nFe_matrix_surf = [0 for P in Ptestarray]
nFeO_matrix_surf = [0 for P in Ptestarray]
nFeO15_matrix_surf = [0 for P in Ptestarray]

# find O/Fe as function of time for a few assumed CMB conditions for (P, T) 
for j1 in range(len(Ptestarray)):
    # initializing
    totaltimesteps = 0

    t = 0 # [yr]
    t_array = np.array([t]) # [yr]

    OperFe_0 = OperFe_min # starting point
    OperFe_evolve = OperFe_0
    OperFe_evolve_array = np.array([OperFe_evolve]) # store the first O/Fe element

    DeltaIW_array = np.array([]) # empty, we don't know first value yet
    Fe_array_CMB = np.array([])
    FeO_array_CMB = np.array([])
    FeO15_array_CMB = np.array([])
    Fe_array_surf = np.array([])
    FeO_array_surf = np.array([])
    FeO15_array_surf = np.array([])

    while t < max_time:
        n_FeO15_guess = N*(OperFe_evolve-n_FeO_guess)/1.5 # with no FeO, we need this much FeO1.5 to reach the desired O/Fe ratio
        n_Fe_guess = N - n_FeO_guess - n_FeO15_guess # the rest is Fe(met)

        res = equilibrium_finder(nFe=n_Fe_guess, nFeO=n_FeO_guess, nFeO15=n_FeO15_guess, KIW=KIW_test_array[j1], KFMQ=KFMQ_test_array[j1], OperFe=OperFe_evolve)
        n_list = res[:3] # first three entries are nFe_eq, nFeO_eq, nFeO1.5_eq
        Fe_array_CMB = np.append(Fe_array_CMB, n_list[0])
        FeO_array_CMB = np.append(FeO_array_CMB, n_list[1])
        FeO15_array_CMB = np.append(FeO15_array_CMB, n_list[2])

        # surface equilibrium
        res_surf = equilibrium_finder(nFe=n_Fe_guess, nFeO=n_FeO_guess, nFeO15=n_FeO15_guess, KIW=KIW_test_array[0], KFMQ=KFMQ_test_array[0], OperFe=OperFe_evolve)
        n_list_surf = res_surf[:3] # first three entries are nFe_eq, nFeO_eq, nFeO1.5_eq at the surface
        Fe_array_surf = np.append(Fe_array_surf, n_list_surf[0])
        FeO_array_surf = np.append(FeO_array_surf, n_list_surf[1])
        FeO15_array_surf = np.append(FeO15_array_surf, n_list_surf[2])

        # compute surface DeltaIW via surface activities
        a_Fe_surf, a_FeO_surf, a_FeO15_surf = n_to_activity(n_list_surf[0], n_list_surf[1], n_list_surf[2], OperFe=OperFe_evolve)
        fO2_surf_FMQ = (a_FeO15_surf/(a_FeO_surf*KFMQ_test_array[0]))**4
        
        DeltaIW = 2*np.log10(a_FeO_surf/a_Fe_surf) # works if a_Fe is defined
        DeltaIW_surf = 4*np.log10(a_FeO15_surf/a_FeO_surf) - 4*np.log10(KFMQ_test_array[0]) + 2*np.log10(KIW_test_array[0])
        DeltaIW_array = np.append(DeltaIW_array, DeltaIW_surf) 
        
        Fe_per_FeT = n_list[0]/sum(n_list) # Fe(met)/FeT at CMB. Molecular component fraction of Fe(met)
        tau_sedi = 1/(k_exponent[j1]*Fe_per_FeT)

        # Don't overstep in time
        # dt = min(tau_sedi*timestepfactor, max_timestep, max_time-t) # Alternative 1: Fixed max_timestep
        dt = min(tau_sedi*timestepfactor, t+max_timestep, max_time-t) # Alternative 2: Relative max_timestep. Allow doubling t (plus the max_timestep)
        
        OperFe_evolve = OperFe_evolve*np.exp(dt/tau_sedi)
        # or equivalently, due to our adaptive time step, OperFe_evolve = OperFe_evolve*np.exp(timestepfactor)
        
        OperFe_evolve_array = np.append(OperFe_evolve_array, OperFe_evolve)
        
        t += dt
        t_array = np.append(t_array, t)
        # print('dt =', dt, 'years', 't = ', t, 'years')

        totaltimesteps += 1
        # store OperFe as function of t
        if t >= max_time:
            OperFe_matrix[j1] = OperFe_evolve_array
            t_matrix[j1] = t_array 
            totaltimesteps_array[j1] = totaltimesteps
            DeltaIW_matrix[j1] = DeltaIW_array
            
            nFe_matrix_CMB[j1] = Fe_array_CMB
            nFeO_matrix_CMB[j1] = FeO_array_CMB
            nFeO15_matrix_CMB[j1] = FeO15_array_CMB

            nFe_matrix_surf[j1] = Fe_array_surf
            nFeO_matrix_surf[j1] = FeO_array_surf
            nFeO15_matrix_surf[j1] = FeO15_array_surf

# print([OperFe_matrix[i][-1] for i in range(len(Ptestarray))])

# %%% Plot the results

# Plot the evolution of O/Fe
plt.figure()
for j1 in range(len(Ptestarray)):
    plt.semilogx(t_matrix[j1], OperFe_matrix[j1], color='k', linestyle=linestylearray[j1])
plt.xlabel('time [years]')
plt.ylabel('O/Fe')
plt.title('Fe sedimenting from the magma ocean')
plt.axhline(1, alpha=0.2, color='k')
PT_legend = plt.legend(handles=PT_handles, title='Assumed CMB\nconditions')

# Plot the evolution of DeltaIW
plt.figure()
for j1 in range(len(Ptestarray)):
    plt.semilogx(t_matrix[j1][:-1], DeltaIW_matrix[j1], color='k', linestyle=linestylearray[j1])
plt.xlabel('time [years]')
plt.ylabel('$\Delta$IW at the magma ocean surface\nwith (P, T) = (0, 3013 K)')
plt.title('CHANCE TO BE MISLEADING: Fe sedimenting from the magma ocean')
PT_legend = plt.legend(handles=PT_handles, title='Assumed CMB\nconditions', loc=(0.01,0.35))

IWminus2handle = plt.axhline(-2, alpha=0.2, color='k', label='Core-mantle separation')
FMQhandle = plt.axhline(4, alpha=0.2, color='tab:blue', label='Modern MORB')
fugacity_legend = plt.legend(handles=[IWminus2handle, FMQhandle], title='Fugacities', loc=(0.01,0.15))
plt.gca().add_artist(PT_legend)

# ax = plt.gca()
# secax = ax.secondary_yaxis(
#     'right',
#     functions=(
#         lambda deltaIW: deltaIW-4,  # deltaIW -> deltaFMQ
#         lambda deltaFMQ: deltaFMQ+4 # deltaFMQ -> deltaIW
#     )
# )
# secax.set_ylabel('$\Delta$FMQ')

# Speciation as function of time for planet with p_CMB, T_CMB = 75 GPa, 3969 K
plt.figure()
plt.semilogx(t_matrix[-1][:-1], nFe_matrix_CMB[j1], color=speciescolors[0], linestyle=linestylearray[-1])
plt.semilogx(t_matrix[-1][:-1], nFeO_matrix_CMB[j1], color=speciescolors[1], linestyle=linestylearray[-1])
plt.semilogx(t_matrix[-1][:-1], nFeO15_matrix_CMB[j1], color=speciescolors[2], linestyle=linestylearray[-1])
# plt.semilogx(t_matrix[-1][:-1], nFe_matrix_surf[j1], color=speciescolors[0], linestyle=linestylearray[0])
# plt.semilogx(t_matrix[-1][:-1], nFeO_matrix_surf[j1], color=speciescolors[1], linestyle=linestylearray[0])
# plt.semilogx(t_matrix[-1][:-1], nFeO15_matrix_surf[j1], color=speciescolors[2], linestyle=linestylearray[0])

# surf_handle, = plt.plot([],[],color='k',linestyle=linestylearray[0], label='surf')
# CMB_handle, = plt.plot([],[],color='k',linestyle=linestylearray[-1], label='CMB')
# surfCMBhandles = [surf_handle, CMB_handle]
# surfCMBlegend = plt.legend(handles=surfCMBhandles)

plt.xlabel('time [years]')
plt.ylabel('Component mole fraction')
plt.title('Fe sedimenting from the magma ocean, P=75GPa case')
species_legend = plt.legend(handles=species_handles, title='Species', loc=(0.01, 0.1))


# %%%
# When we don't have Fe(met), we can still define, 
# through the FeO + 1/4 O2 <-> FeO1.5 reaction a
# DeltaIW = 4*log10(a_FeO1.5/a_FeO) - 4 log10(K_FMQ) + 2 log10(K_IW)
# Compute the equilibrium constant dependent terms 4 log10(K_FMQ) + 2 log10(K_IW)
# for some different temperatures

Tarray = np.linspace(1400, 4000) # K
Parray = 0.0001*np.zeros_like(Tarray) # [GPa], Psurf = 1 bar standard state
KIW_lin = np.array([iron_wustite_K(P, T) for P, T in zip(Parray, Tarray)])
KFMQ_lin = np.array([Fe3plusreact_K(P, T) for P, T in zip(Parray, Tarray)])
logKdifference = 2*np.log10(KIW_lin)-4*np.log10(KFMQ_lin)

plt.figure()
plt.plot(Tarray, logKdifference, '.')
plt.title("$\Delta$IW-$\Delta$'FMQ' at P = 1 bar\n2log$_{10}$(K$_{\mathrm{IW}}$)-4log$_{10}$(K$_{\mathrm{'FMQ'}}$)")
plt.xlabel('Temperature [K]')
plt.ylabel('[dex]')

# Also plot DeltaIW as function of P, T for a fixed Fe3+/FeT
# Fe3+/FeT = 0.02 and 0.06 should correspond to IW+3 and IW+5
# according to Hirschmann2022 introduction
Fe3perFeT_Hirschmann1 = 0.02 
Fe3perFeT_Hirschmann2 = 0.06
DeltaIW_Hirschmann1 = 4*np.log10(Fe3perFeT_Hirschmann1)+logKdifference # for different T
DeltaIW_Hirschmann2 = 4*np.log10(Fe3perFeT_Hirschmann2)+logKdifference # for different T

plt.figure()
plt.plot(Tarray, DeltaIW_Hirschmann1, label=f'Fe3+/FeT={Fe3perFeT_Hirschmann1:.2f}')
plt.plot(Tarray, DeltaIW_Hirschmann2, label=f'Fe3+/FeT={Fe3perFeT_Hirschmann2:.2f}')
plt.axhline(3,color='k', alpha=0.2)
plt.axhline(5,color='k', alpha=0.2)
plt.title('DeltaIW from Hirschmann2022\nhow does it not line up for any T?')
plt.xlabel('Temperature [K]')
plt.ylabel('DeltaIW computed at P = 0 \nfrom specified Fe3+/FeT and known KIW, KFMQ')
plt.legend()


# Also plot DeltaIW as a function of Fe3+/FeT for a fixed P, T

# single (P, T)_surf
Psurf_test, Tsurf_test = 1e5*1e-9, 2173 # 100 kPa in [GPa], 2173 [K] surface temp. 
KIW_surf_test = iron_wustite_K(Psurf_test, Tsurf_test)
KFMQ_surf_test = Fe3plusreact_K(Psurf_test, Tsurf_test) 
# stoichiometry_Fe3plus = 4 # Self-consistent & natural choice. FeO + 1/4 O2 <-> FeO1.5
stoichiometry_Fe3plus = 1/0.1917 # Factor in front of fO2 != 1/4 for Hirschmann2022. Instead = 1/0.1917

# Trying to match Hirschmann2022 text 
# "For a terrestrial magma ocean with Fe3+/FeT ratios of 0.034–0.103 (Fig. 7),
# the corresponding fO2 ranges from IW−1.2 to IW+1.4 using Eq. (21) from this study"
# Fe3plus_lin = np.logspace(np.log10(0.034), np.log10(0.103), 20) # Fe3+/FeT

# Or use a range of values to plot Fig. 9
Fe3plus_lin = np.logspace(-2.5, np.log10(0.6), 20) # Match Fig. 9. Fe3+/FeT

Fe3plusperFe2plus_lin = Fe3plus_lin*(1-Fe3plus_lin)**(-1) # Fe3+/Fe2+ = Fe3+/FeT * (1-Fe3+/FeT)^-1
DeltaIWwrtFe3plus = 4*np.log10(Fe3plusperFe2plus_lin)-4*np.log10(KFMQ_surf_test)+2*np.log10(KIW_surf_test) 
DeltaIWwrtFe3plus_2 = stoichiometry_Fe3plus*np.log10(Fe3plusperFe2plus_lin)-stoichiometry_Fe3plus*np.log10(KFMQ_surf_test)+2*np.log10(KIW_surf_test) 

Fe3plusperFe2plus_lin_Sossi = 10**(0.252*DeltaIWwrtFe3plus-1.4) # Sossi2020 corrected regression in Hirschmann2022 (Eq. 34)
Fe3plus_lin_Sossi = Fe3plusperFe2plus_lin_Sossi/(1+Fe3plusperFe2plus_lin_Sossi) # backwards conversion 

# Sossi2020 datapoints, Table S4. 
Fe3plusperFeT_Sossi_data = np.array([0.014, 0.019, 0.045, 0.124, 0.172, 0.217, 0.240, 0.267, 0.226, 0.269, 0.277, 0.440])
log10fO2_Sossi_data = np.array([-7.86, -6.74, -5.32, -3.92, -3.00, -2.72, -2.72, -2.72, -2.58, -2.53, -2.49, 0.00])
DeltaIW_Sossi_data = log10fO2_Sossi_data + 2*np.log10(KIW_surf_test) # log10(fO2) = DeltaIW - 2 log10(KIW)

# Trying to match Hirschmann2022 fig. 9
plt.figure()
plt.plot(DeltaIWwrtFe3plus, Fe3plus_lin, color='k', label='1/4 fO2 enforced')
plt.plot(DeltaIWwrtFe3plus_2, Fe3plus_lin, color='b', label='Hirschmann2022, Eq. 21')
plt.plot(DeltaIWwrtFe3plus, Fe3plus_lin_Sossi, linestyle='dotted', linewidth=2, color='r', label='Sossi2020, corrected')
plt.plot(DeltaIW_Sossi_data, Fe3plusperFeT_Sossi_data, 'o', color='k', mfc='white', markersize=10, label='Sossi2020 data')
plt.xlabel('$\Delta$IW, '+IWpaper)
plt.ylabel('Fe$^{3+}$/Fe$^{\mathrm{T}}$')
plt.title('Hirschmann2022 conversion from\nspeciation to $\Delta$IW at $(P,T)$=(1 bar, 2173 K)')
plt.xlim(left=-2, right=+6.3)
plt.ylim(bottom=0, top=0.6)
plt.legend()

# %%% Compare KIW computed from Komabayashi2014 and parametrized by Hirschmann2021

# DeltaIW wrt fO2 for different IW papers for a single (P, T)
log10fO2_lin = np.linspace(-10, -2, 20)
KIW_Hirschmann_test = iron_wustite_K_Hirschmann(Psurf_test, Tsurf_test)
KIW_Komabayashi_test = iron_wustite_K_Komabayashi(Psurf_test, Tsurf_test)
plt.figure()
plt.plot(log10fO2_lin, log10fO2_lin+2*np.log10(KIW_Hirschmann_test), color='b', label='Hirschmann2021 IW')
plt.plot(log10fO2_lin, log10fO2_lin+2*np.log10(KIW_Komabayashi_test), color='r', label='Komabayashi2014 IW*')
plt.xlabel('log$_{10}$(f$_{\mathrm{O}_2}$)')
plt.ylabel('$\Delta$IW')
plt.title('Comparing $\Delta$IW and $\Delta$IW*\n at (P, T) = (1 bar, 2173 K)')
plt.legend()

# KIW varying (P, T)
KIW_Hirschmann_test_array = np.zeros(len(Plin))
KIW_Komabayashi_test_array = np.zeros(len(Tlin))
for j, (Ptest, Ttest) in enumerate(zip(Plin, Tlin)):
    KIW_Hirschmann_test_array[j] = iron_wustite_K_Hirschmann(Ptest, Ttest)
    KIW_Komabayashi_test_array[j] = iron_wustite_K_Komabayashi(Ptest, Ttest)

plt.figure()
plt.semilogy(Plin, KIW_Hirschmann_test_array, color='b', label='Hirschmann2021, solid')
plt.semilogy(Plin, KIW_Komabayashi_test_array, color='r', label='Komabayashi2014, liquid')
plt.xlabel('Pressure [GPa]')
plt.ylabel('K$_\mathrm{IW}$')
plt.title('IW-buffer comparison')
plt.xlim(left=0, right=max(Plin))

ax = plt.gca()
secax = ax.secondary_xaxis(
    'top',
    functions=(
        lambda P: np.interp(P, Plin[::-1], Tlin[::-1]),  # P -> T
        lambda T: np.interp(T, Tlin[::-1], Plin[::-1])   # T -> P
    )
)
secax.set_xlabel('T [K]')
secax.set_xlim(left=min(Tlin), right=max(Tlin))
ticks = secax.get_xticks()
secax.set_xticks(ticks[ticks <= Tlin.max()])
plt.legend()


# %%% Testing the activity coefficient calc. of Hirschmann2022

# melt composition parameters
y1=-1198.4/2.303 # 2.303 factor includes conversion from ln (gamma_3+/gamma_2+) -> log10 (gamma_3+/gamma_2+)
y2= -426.82/2.303 
y3= 1138.371/2.303
y4= 4232.933/2.303
y5= 6650.972/2.303
y6= 7998.434/2.303
y7= -10298.6/2.303
y8= -2866.92/2.303
y9= -2663.74/2.303

