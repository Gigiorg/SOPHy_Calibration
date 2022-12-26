# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:13:31 2022

@author: GIBS
"""

#%%

import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

h_path = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\EXP2\Table_exp2.xlsx'
v_path = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\EXP5\Table_exp5.xlsx'

#Constants

rEsf = 0.1765
opFreq = 9.345e9 
c = 3e8 


pT1 = 200*(0.1/400)
pT2 = 1.91

gT = gR = 10**(38.5/10)
lambdaRadar = round(c/opFreq, 3)      
sigma = np.pi*rEsf**2
gLNA1 = 10**(30/10)
gLNA2 = 10**(40/10)



#Horizontal

df_h = pd.ExcelFile(h_path).parse('hits')


rwf_h = np.array(df_h.RWF)
bwf_h = np.array(df_h.BWF)
rangeT_h = np.array(df_h.range)
rPower_h = np.array(df_h["R Power [W]"])

sqLosses_h = ((pT1*gT*gR*gLNA1*gLNA2*lambdaRadar**2)/((4*np.pi)**3*rangeT_h**4*rPower_h)) *sigma*rwf_h*bwf_h
sqLosses_h_avg = sqLosses_h.mean()



#Vertical


df_v = pd.ExcelFile(v_path).parse('hits')


rwf_v = np.array(df_v.RWF)
bwf_v = np.array(df_v.BWF)
rangeT_v = np.array(df_v.range)

rPower_v = np.array(df_v["R Power [W]"])

sqLosses_v = ((pT1*gT*gR*gLNA1*gLNA2*lambdaRadar**2)/((4*np.pi)**3*rangeT_v**4*rPower_v)) *sigma*rwf_v*bwf_v
sqLosses_v_avg = sqLosses_v.mean()