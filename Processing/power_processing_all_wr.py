# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 08:05:15 2022

@author: GIBS
"""

#%%

import pandas as pd
import os
import datetime
import numpy as np
import h5py as h5
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
import math
from scipy.signal import butter, lfilter, freqz


PATH_1 = r'C:\Users\GIBS\Documents\Experimentos\Exp1_Final\param-0.5km-0\2022-05-07T17-00-00'
PATH_2 = r'C:\Users\GIBS\Documents\Experimentos\Exp2_Final\param-0.5km-0\2022-05-08T08-00-00'
PATH_3 = r'C:\Users\GIBS\Documents\Experimentos\Exp3_Final\param-0.5km-0\2022-05-08T17-00-00'
PATH_4 = r'C:\Users\GIBS\Documents\Experimentos\Exp4_Final\param-0.5km-0\2022-05-08T17-00-00'
PATH_5 = r'C:\Users\GIBS\Documents\Experimentos\Exp5_Final\param-0.5km-0\2022-05-09T08-00-00'
PATH_6 = r'C:\Users\GIBS\Documents\Experimentos\Experimento6\param-0.5km-0\2022-05-09T10-00-00'

POS_1 = r'C:\Users\GIBS\Documents\Experimentos\Exp1_Final\table_1_end.csv'
POS_2 = r'C:\Users\GIBS\Documents\Experimentos\Exp2_Final\table_2_end.csv'
POS_3 = r'C:\Users\GIBS\Documents\Experimentos\Exp3_Final\table_3_end.csv'
POS_4 = r'C:\Users\GIBS\Documents\Experimentos\Exp4_Final\table_4_end.csv'
POS_5 = r'C:\Users\GIBS\Documents\Experimentos\Exp5_Final\table_5_end.csv'
POS_6 = r'C:\Users\GIBS\Documents\Experimentos\Experimento6\posiciones_exp6_f3.csv'

PATHS = [PATH_1, PATH_2, PATH_3, PATH_4, PATH_5, PATH_6]

PATHS_TAB = [POS_1, POS_2, POS_3, POS_4, POS_5, POS_6]

PATH_ALL = r'C:\Users\GIBS\Documents\Experimentos\Exp5_Final\Drone\plots_drone' 

dias = [7,8,8,8,9,9]  # Day when the experiment was realized
d_el = [5,5,7,7,7,7]  # Delay in EL associated to certain experiment
exps = {}


count = 1

#Iterate over each experiment
for exp in PATHS:
    power_files = []
    muestras = {}
    
    #Get all the h5 files from an experiment
    for root, directories, file in os.walk(exp):
        for file in file:
            if(file.endswith(".hdf5")):
                power_files.append(file)
                file_path = exp+'\\'+file
                h5a = h5.File(file_path, 'r')
                utctime = h5a['Data']['time'][0]
                datetime2 = datetime.datetime.fromtimestamp(utctime)
            
                if count <= 3:
                    Data_Arr_H = 10*np.log10(h5a['Data']['data_param']['channel00'][:,-33:])  #0m
                    Data_Arr_H[0:71,:10] = -55.0
                    Data_Arr_H[Data_Arr_H > -11] = -55.0
                    
        
                
                elif (count == 4) or (count == 5):
                    Data_Arr_H = 10*np.log10(h5a['Data']['power']['H'][:,-33:])  #0m
                    Data_Arr_H[0:51,:10] = -55.0
                    Data_Arr_H[Data_Arr_H > -11] = -55.0
                    
                
                else:
                
                    Data_Arr_H = 10*np.log10(h5a['Data']['data_param']['channel00'][:,-33:])  #0m
                    Data_Arr_H[0:71,:10] = -55.0
                    Data_Arr_H[Data_Arr_H > -11] = -55.0
                    
            
                
                muestras.update({file:{"time":datetime2.strftime( "%Y-%m-%d  %H:%M:%S"),
                                       "elevation":list(h5a['Metadata']['elevation']),
                                       "azimuth":list(h5a['Metadata']['azimuth']),
                                       "range":list(h5a['Metadata']['range']),
                                       "powerH":Data_Arr_H}})

    exps.update({count:muestras})
    count += 1

#Iterate over each experiment
for exp in range(1,7):
    
    #Iterate over each file/sample
    for i in list(exps[exp].keys()):
        profiles_H = []
        profiles_H_fxd = []
        powerH = exps[exp][i]['powerH']
        
        #Iterate over each row from the power dataset of the sample 
        for j in range(powerH.shape[0]):
            if (len(np.where(powerH[j] > -40.0)[0])!= 0 and powerH[j][10]<-25.):
                
                profiles_H.append([exps[exp][i]['elevation'][j], list(np.where(powerH[j] > -40.0)[0])])
                profiles_H_fxd.append([exps[exp][i]['elevation'][j-d_el[exp-1]], list(np.where(powerH[j] > -40.0)[0])])
                
            else:
                profiles_H.append([exps[exp][i]['elevation'][j], []])
                profiles_H_fxd.append([exps[exp][i]['elevation'][j-d_el[exp-1]], []])
        
        exps[exp][i].update({"profiles_H":profiles_H})
        exps[exp][i].update({"profiles_H_fxd":profiles_H_fxd})
        
        count = 0
        drone = {}
        esfera = {}
        rpower_sphere = []
        rpower_drone = []

        #Separation of power samples associated to the sphere / UAV
        
        for j in range(len(exps[exp][i]["profiles_H"])):
            
            if count == 0 :
                if(len(exps[exp][i]["profiles_H"][j][1]) != 0):
                    for k in exps[exp][i]["profiles_H"][j][1]:
                        drone.update({str(exps[exp][i]["profiles_H"][j][0]) + " " + str(k) : [exps[exp][i]["profiles_H"][j][0],
                                                                                          exps[exp][i]["profiles_H"][j-d_el[exp-1]][0], 
                                                                                         k, 
                                                                                         j, 
                                                                                         10**(exps[exp][i]['powerH'][j][k]/10)]})

                    count = 1
                    
            if count == 1:
                if(len(exps[exp][i]["profiles_H"][j][1]) != 0):
                    for k in exps[exp][i]["profiles_H"][j][1]:
                        drone.update({str(exps[exp][i]["profiles_H"][j][0]) + " " + str(k) : [exps[exp][i]["profiles_H"][j][0],
                                                                                         exps[exp][i]["profiles_H"][j-d_el[exp-1]][0], 
                                                                                         k, 
                                                                                         j, 
                                                                                         10**(exps[exp][i]["powerH"][j][k]/10)]})
                        
                else:
                    count = 2
                    
            if count == 2:
                if(len(exps[exp][i]["profiles_H"][j][1]) != 0):
                    count = 3
            
            if count == 3:
                if(len(exps[exp][i]["profiles_H"][j][1]) != 0):
                    for k in exps[exp][i]["profiles_H"][j][1]:    
                        esfera.update({str(exps[exp][i]["profiles_H"][j][0]) + " " + str(k) : [exps[exp][i]["profiles_H"][j][0],
                                                                                          exps[exp][i]["profiles_H"][j-d_el[exp-1]][0], 
                                                                                          k, 
                                                                                          j, 
                                                                                          10**(exps[exp][i]["powerH"][j][k]/10)]})
                        rpower_sphere.append(10**(exps[exp][i]["powerH"][j][k]/10))
                        
        exps[exp][i].update({"esfera":esfera})
        exps[exp][i].update({"drone":drone})
        exps[exp][i].update({"rpower_sph":rpower_sphere})
        exps[exp][i].update({"rpower_dro":rpower_drone})
        
        #Updating the dict with the maximum received power associated to the sphere 
        if len(rpower_sphere) != 0:    
            exps[exp][i].update({"maxpower_sph":max(rpower_sphere)})
        else:                                                                                                 
            exps[exp][i].update({"maxpower_sph":0})
        
#%%

pulse_width = 0.1e-6
c = 3e8
sigma_r = 0.35*pulse_width*c/2
sigma_xy = 1.8/2.36
r_esf = 0.1765                                             # Sphere radius [m]
rcs = round(np.pi*r_esf**2, 2)                             # Radar Cross Section (Optic Region)
freq_oper = 9.345e9                                        # Radar operative frequency
w_length = round(c/freq_oper, 3)                           # Wavelength of the rad
antenna_gain = 10**(38.5/10)                               # Antenna Gain [dBi]
k_m = 0.93                                                 # Atmospheric refractive index 
pulse_width = 0.1e-6                                       # Pulse width of the transmitted pulses [s]
beam_width = np.deg2rad(1.8)                               # Beam width of the transmitting antenna
t_power = 1.91                                             # Transmitted Power
WG_len = 7.87                                              # Waveguide length (2.4m) [ft]
 #Perdidas
alfa = 1.4e-2*0.2                                          # Atmospheric attenuation
L_circu = 0.5                                              # Circulator losses
L_rot_joint = 0.6                                          # Rotary Joint losses
L_wg = 0.1*WG_len                                          # Waveguide losses
L_adap = 0.5                                               # SMA-WR90 adapter losses
Glna = 10**(83.5/10) 

#L_total = 10**( (alfa + L_circu + L_rot_joint + L_wg + L_adap)/10)     # Total losses
L_total = 10**( (2*alfa + 2*L_circu + 2*L_wg + 4*L_adap)/10)     # Total losses
L_total_db = 2*alfa + 2*L_circu + 2*L_wg + 4*L_adap

def calc_constante():
        
    C_sph = (t_power*(antenna_gain**2)*(Glna)*(w_length**2))/(((4*np.pi)**3)*(L_total**2))
    return C_sph

#Iterate over each experiment
for exp in range(1,7):
     
    l_28 = {} 
    l_29 = {} 
    l_30 = {}
    
    df = pd.read_csv(PATHS_TAB[exp-1])
    
    date = []
    l_esf = df.l_e
    h_esf = df.z_e
    constant = []
    time_W = []
    power = []
    power_db = []
    azimuth = []
    file = []
    constant_db = []
    range_r = []
    ro_max  =[]
    wr = []
    c_after = []
    for i in df.time:
        
        dat, tim = i.split(" ")
        hh,mm,ss = tim.split(":")
        time_base = datetime.datetime(2022, 5, dias[exp-1], int(hh), int(mm), int(ss))
        date.append(time_base.strftime( "%Y-%m-%d  %H:%M:%S"))
    

    #Iterate over each file/sample
    for i in list(exps[exp].keys()):
        utctime = exps[exp][i]['time']
        try:
            
            #Calculate the radar calibration constant...
            idx_sph = date.index(utctime)
            h = h_esf[idx_sph]
            l = l_esf[idx_sph]
            r = np.sqrt((float(h)-2.9)**2 + float(l)**2)
            p_max =  exps[exp][i]['maxpower_sph']
            C = (p_max*(r**4))/rcs
            
            #
            r_max_idx = exps[exp][i]['rpower_sph'].index(max(exps[exp][i]['rpower_sph']))
            d_sph = list(exps[exp][i]['esfera'].keys())[r_max_idx]
            range_max = exps[exp][i]['esfera'][d_sph][2]
            perf_max = exps[exp][i]['esfera'][d_sph][1]
            #print(perf_max)
            v_max = 15*range_max-15
            
            #print(C)
            constant.append(C)
            constant_db.append(10*np.log10(C))
            power.append(p_max)
            power_db.append(10*np.log10(p_max))
            azimuth.append(float(i[-11:-7]))
            time_W.append(utctime)
            file.append(i)
            range_r.append(r)
            ro_max.append(15*range_max-15)
            Wr = np.exp(-((r-v_max)**2)/(2*sigma_r**2))
            wr.append(Wr)
            c_after.append(10*np.log10((p_max*(r**4))/(rcs*Wr)))
        except:
            idx_sph = 0
    
    
    data = [file, time_W, azimuth, range_r, ro_max, power, power_db, constant, constant_db, wr, c_after]
    
    #Print all the calculated results for each sample in an excel table
    df = pd.DataFrame(data)
    df = df.transpose()
    df.columns = ["Filename", "Time", "Azimuth", "Range","ro Max", "Received Power", "R Power [dB]","Constant", "Constant [db]", "Wr",
                  "Constant after [dB]"]
    df.to_excel(r'C:\Users\GIBS\Documents\Experimentos\Experiments_wr'+str(exp)+'.xlsx', sheet_name='tabla')
    
    
    
#%%
for i in PATHS_TAB:
    a = pd.read_csv(i)
    print(a.time[0])
          