# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 10:38:27 2022

@author: GIBS
"""

import numpy as np

#SOPHy calibration variables

losses_h = 0.14483353615030312  #Estimated losses for the H transmitting line
losses_v = 0.27325983910736124  #Estimated losses for the V transmitting line

rEsf = 0.1765                   #Sphere radius
opFreq = 9.345e9                #SOPHy operative frequency
c = 3e8                         #Speed of light
p_width = 0.1e-6 # REPLACE WITH THE CURRENT OPERATIVE PULSE WIDTH!

pT = 200*(0.1/400)              #Transmitted Power for Experiments

G = gT = gR = 10**(38.7/10)     #Antenna Gain
lambdaRadar = round(c/opFreq, 3)#Wavelength       
sigma = np.pi*rEsf**2           #Sphere Radar Cross Section
gLNA_sp = 10**(70/10)           #LNA gain for the experiments
theta = np.deg2rad(1.98)        #Antenna 3dB beamwidth
k_m = 0.93                      #Refractive index of the environment


#Spherical constant

c_sph_h = (G**2 * lambdaRadar**2 )/(((4*np.pi)**3) * losses_h)
c_sph_v = (G**2 * lambdaRadar**2 )/(((4*np.pi)**3) * losses_v)

#Reflectivity constant

c_z_h = (16 * np.log(2) * lambdaRadar**2 * 10**18 )/(c_sph_h * c * p_width * np.pi * theta**2 * np.pi**5 * k_m)
c_z_v = (16 * np.log(2) * lambdaRadar**2 * 10**18 )/(c_sph_v * c * p_width * np.pi * theta**2 * np.pi**5 * k_m)

c_z_h_db = 10*np.log10(c_z_h)
c_z_v_db = 10*np.log10(c_z_v)


### CALIBRATION FOR REFLECTIVITY

r = 2                 # Range [Km]
gLNA_ac = 10**(51/10) # LNA Gain
pR = 0.01             # Received Power
pT = 1                # Transmitted Power [W]


# Horizontal Z
Z1_h_db = 10*np.log10(1000*pR) + 20*np.log10(r) + c_z_h_db - 10*np.log10(1000*pT)- 10*np.log10(gLNA_ac)

Z2_h = (pR * r**2  * c_z_h )/(pT * gLNA_ac)
Z2_h_db = 10*np.log10(Z2_h)

# Vertical Z

Z1_v_db = 10*np.log10(1000*pR) + 20*np.log10(r) + c_z_v_db - 10*np.log10(1000*pT)- 10*np.log10(gLNA_ac)

Z2_v = (pR * r**2  * c_z_v )/(pT * gLNA_ac)
Z2_v_db = 10*np.log10(Z2_v)

