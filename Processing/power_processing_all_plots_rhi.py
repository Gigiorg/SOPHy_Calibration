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
POS_6 = r'C:\Users\GIBS\Documents\Experimentos\Exp6_Final\table_6_end.csv'


SAVE_PATH_1 = r'C:\Users\GIBS\Documents\Experimentos\Plots_exp_1'
SAVE_PATH_2 = r'C:\Users\GIBS\Documents\Experimentos\Plots_exp_2'
SAVE_PATH_3 = r'C:\Users\GIBS\Documents\Experimentos\Plots_exp_3'
SAVE_PATH_4 = r'C:\Users\GIBS\Documents\Experimentos\Plots_exp_4'
SAVE_PATH_5 = r'C:\Users\GIBS\Documents\Experimentos\Plots_exp_5'
SAVE_PATH_6 = r'C:\Users\GIBS\Documents\Experimentos\Plots_exp_6'



SAVE_RHI_1 = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\RHI\Plots_exp_1'
SAVE_RHI_2 = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\RHI\Plots_exp_2'
SAVE_RHI_3 = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\RHI\Plots_exp_3'
SAVE_RHI_4 = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\RHI\Plots_exp_4'
SAVE_RHI_5 = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\RHI\Plots_exp_5'
SAVE_RHI_6 = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\RHI\Plots_exp_6'



PATHS = [PATH_1, PATH_2, PATH_3, PATH_4, PATH_5, PATH_6]

PATHS_TAB = [POS_1, POS_2, POS_3, POS_4, POS_5, POS_6]


PATH_ALL = r'C:\Users\GIBS\Documents\Experimentos\Exp5_Final\Drone\plots_drone' 

PATH_PLOTS = [SAVE_PATH_1, SAVE_PATH_2, SAVE_PATH_3, SAVE_PATH_4, SAVE_PATH_5, SAVE_PATH_6]
PATH_RHI = [SAVE_RHI_1, SAVE_RHI_2, SAVE_RHI_3, SAVE_RHI_4, SAVE_RHI_5, SAVE_RHI_6]
dias = [7,8,8,8,9,9]  # Day when the experiment was realized
d_el = [5,5,7,7,7,7]  # Delay in EL associated to certain experiment
OFF = 1 # 15m
exps = {}


count = 1

def getdataset_Exp(PATH):
    
    """
    Obtain the most important columns in the dataset from a certain hdf5 file for an experiment.
    
    Arguments:
        
             PATH: path of a certain hdf5 file (string)
        
    
    Returns:
         
        a list with the most import columns from the experiment
        
        a[0]: datetime series
        a[1]: height of the drone series 
        a[2]: distance from the radar to the drone series 
        a[3]: height of the sphere series 
        a[4]: distance from the radar to the sphere series 
        a[5]: distance from the radar to the drone within the x component series 
        a[6]: distance from the radar to the drone within the y component series 
        a[7]: distance from the drone to the sphere within its relative x component series 
        a[8]: distance from the drone to the sphere within its relative y component series 
        
    """
    
    
    time_list = []
    df = pd.read_csv(PATH)
    
    for x in df.time:
        a = x.split("/")
        year = a[2].split(" ")[0]
        hour = a[2].split(" ")[1]
        month = a[1]
        day = a[0]
        
        po = year+"-"+month+"-"+day+" "+hour
        time_list.append(datetime.datetime.strptime(po, "%Y-%m-%d %H:%M:%S"))
        
    return [time_list,df.z_d, df.l_d, df.z_e, df.l_e, df.x_d, df.y_d, df.E_l, df.F_l, df.roll, df.pitch]



def get_coords(ang, rangearr):    
    
    """
    Get the s-h cartesian coordinates of the center of the resolution cell using the elevation 
    profile and the range cell.
    
    Arguments:
        
             ang: profile of the pedestal [??] (float)
        rangearr: range cell of the radar as the index from the range array (integer)
        
        
    Returns:
         
        a list with the coordinates of the estimated resolution cell
        
        a[0]: lower left corner of the rectangle
        a[1]: height of lower side (horizontal) of the resolution cell (0 - height) 
        a[2]: height of the resolution cell volume (upper height - lower height)
        a[3]: height of the center of the resolution cell
        
    """
    
    center = 15*rangearr
    first_range = center*np.cos(np.deg2rad(ang))
    alt_centro = center*np.sin(np.deg2rad(ang))
    alt_sup = center*np.sin(np.deg2rad(ang+0.9))
    alt_inf = center*np.sin(np.deg2rad(ang-0.9))
    
    dif_alt = alt_sup - alt_inf
        
    a = [center - 7.5, alt_inf, dif_alt, alt_centro]

    return a

def get_coords(ang, rangearr):    
    
    first_range = (15*rangearr)*np.cos(np.deg2rad(ang))
    center = first_range
    alt_centro = (15*rangearr)*np.sin(np.deg2rad(ang))
    alt_sup = (15*rangearr)*np.sin(np.deg2rad(ang+0.9))
    alt_inf = (15*rangearr)*np.sin(np.deg2rad(ang-0.9))
    
    dif_alt = alt_sup - alt_inf
        
    a = [center - 7.5, alt_inf, dif_alt, alt_centro]

    return a

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
                    Data_Arr_H = 10*np.log10(h5a['Data']['data_param']['channel00'][:,-33+OFF:])  #0m
                    Data_Arr_H[0:71,:10] = -55.0
                    Data_Arr_H[Data_Arr_H > -11] = -55.0
                    
        
                
                elif (count == 4) or (count == 5):
                    Data_Arr_H = 10*np.log10(h5a['Data']['power']['H'][:,-33+OFF:])  #0m
                    Data_Arr_H[0:51,:10] = -55.0
                    Data_Arr_H[Data_Arr_H > -11] = -55.0
                    
                
                else:
                
                    Data_Arr_H = 10*np.log10(h5a['Data']['data_param']['channel00'][:,-33+OFF:])  #0m
                    Data_Arr_H[0:51,:10] = -55.0
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
    
    
    ####
    time_pos = getdataset_Exp(PATHS_TAB[exp-1])[0]
    drone_h_arr = getdataset_Exp(PATHS_TAB[exp-1])[1]
    drone_s_arr = getdataset_Exp(PATHS_TAB[exp-1])[2]
    
    esf_h_arr = getdataset_Exp(PATHS_TAB[exp-1])[3]
    esf_s_arr = getdataset_Exp(PATHS_TAB[exp-1])[4]
    
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
            
        for j in list(exps[exp][i]['drone'].keys()):
            rpower_drone.append(exps[exp][i]['drone'][j][-1])
            
            
        #Plotting each sample with the estimated position of the sphere and the max volume
        fig, ax = plt.subplots(1,1,figsize=(15,15))
        
        #Timestamp from the h5 file
        date2, time2 = exps[exp][i]['time'].split("  ")
        hh,mm,ss = time2.split(":")
        time_base = datetime.datetime(2022, 5, dias[exp-1], int(hh), int(mm), int(ss))
        #print(time_base)
         
        #Highest profile (Upper position of the pedestal in RHI mode)
        perf_max = exps[exp][i]['profiles_H'][0][0]
        
       
        
        try:
           
           if( len(exps[exp][i]['drone']) != 0):
               max_power_drone_idx = rpower_drone.index(max(rpower_drone))
               max_drone_refer = list(exps[exp][i]['drone'].keys())[max_power_drone_idx]
               dif_ang_drone = round((perf_max-exps[exp][i]['drone'][max_drone_refer][1])/10,1)
               dif_secs_drone = math.floor(dif_ang_drone)
               time_drone = time_base + datetime.timedelta(seconds = dif_secs_drone)
               idx_drone = list(time_pos).index(time_drone)
               #print(idx_drone)
               plt.plot(drone_s_arr[idx_drone],drone_h_arr[idx_drone]-2.9,'ko', linewidth=2, markersize=10)
                
               #Tiempo referencial para la esfera y ploteo
           if( len(exps[exp][i]['esfera']) != 0):   
               max_power_sph_idx = rpower_sphere.index(max(rpower_sphere))
               max_sph_refer = list(exps[exp][i]['esfera'].keys())[max_power_sph_idx]
               dif_ang_esf = round((perf_max-exps[exp][i]['esfera'][max_sph_refer][1])/10,1)
               dif_secs_esf = math.floor(dif_ang_esf)
               dif_decs_esf = int(round(math.modf(dif_ang_esf)[0],1)*10)
               time_sphere = time_base + datetime.timedelta(seconds = dif_secs_esf)
               idx_sphere = list(time_pos).index(time_sphere)
               idx_sphere_f = int(idx_sphere + 10*dif_decs_esf/2)
               plt.plot(esf_s_arr[idx_sphere_f],esf_h_arr[idx_sphere_f]-2.9,'ko', linewidth=2, markersize=10)
                
           potentials = {}
           potentials_d = {}
           
           for j in exps[exp][i]['drone'].keys():
               
               corr_perf = exps[exp][i]['profiles_H'][ exps[exp][i]['drone'][j][3]-d_el[exp-1]][0]
               range_perf = exps[exp][i]['drone'][j][2]
               #esfera_c = get_coords(j[1],k)
               drone_c = get_coords(corr_perf,range_perf)
               ax.add_patch(Rectangle((drone_c[0],drone_c[1]),15,drone_c[2],
                                      edgecolor = 'black', facecolor = 'lightblue', fill=True))
               plt.plot(drone_c[0] + 7.5,drone_c[3],'xw', markersize = 12)
               power =  exps[exp][i]['powerH'].shape[0]
               potentials_d.update({j:{"err_s":np.abs(drone_c[0] + 7.5 -drone_s_arr[idx_drone]),
                                     "err_h":np.abs(drone_c[3] - (drone_h_arr[idx_drone] -2.9)),
                                     "med_h":drone_c[3]-drone_c[1]}})    
               
           exps[exp][i].update({"meds_drone":potentials_d})
           
           if( len(exps[exp][i]['drone']) != 0): 
               max_power_drone_idx = rpower_drone.index(max(rpower_drone)) 
               max_drone_refer = list(exps[exp][i]['drone'].keys())[max_power_drone_idx]
               max_range =  exps[exp][i]['drone'][max_drone_refer][2]
               max_perf = exps[exp][i]['drone'][max_drone_refer][1]
               drone_max = get_coords(max_perf, max_range)
               ax.add_patch(Rectangle((drone_max[0],drone_max[1]),15,drone_max[2],
                                      edgecolor = 'black', facecolor = 'yellow', fill=True))
               plt.plot(drone_max[0] + 7.5,drone_max[3],'xw', markersize = 12)
              
               
           for j in exps[exp][i]['esfera'].keys():
               
               corr_perf = exps[exp][i]['profiles_H'][ exps[exp][i]['esfera'][j][3]-d_el[exp-1]][0]
               range_perf = exps[exp][i]['esfera'][j][2]
               #esfera_c = get_coords(j[1],k)
               esfera_c = get_coords(corr_perf,range_perf)
               ax.add_patch(Rectangle((esfera_c[0],esfera_c[1]),15,esfera_c[2],
                                      edgecolor = 'black', facecolor = 'red', fill=True))
               plt.plot(esfera_c[0] + 7.5,esfera_c[3],'xw', markersize = 12)
               power =  exps[exp][i]['powerH'].shape[0]
                   
               potentials.update({j:{"err_s":np.abs(esfera_c[0] + 7.5 -esf_s_arr[idx_sphere_f]),
                                     "err_h":np.abs(esfera_c[3] - (esf_h_arr[idx_sphere_f] -2.9)),
                                     "med_h":esfera_c[3]-esfera_c[1]}})
               
           exps[exp][i].update({"meds":potentials})
           
           
           
           if( len(exps[exp][i]['esfera']) != 0): 
               max_power_sph_idx = rpower_sphere.index(max(rpower_sphere)) 
               max_sph_refer = list(exps[exp][i]['esfera'].keys())[max_power_sph_idx]
               max_range =  exps[exp][i]['esfera'][max_sph_refer][2]
               max_perf = exps[exp][i]['esfera'][max_sph_refer][1]
              
               
               esfera_max = get_coords(max_perf, max_range)
               ax.add_patch(Rectangle((esfera_max[0],esfera_max[1]),15,esfera_max[2],
                                      edgecolor = 'black', facecolor = 'green', fill=True))
               
               plt.plot(esfera_max[0] + 7.5,esfera_max[3],'xw', markersize = 12)
           
               
           si_cta = 0
           dr_cta = 0
           
           #Determine the presence of the sphere inside a volume cell
           for x in list(exps[exp][i]['meds'].keys()):
                 error_s = exps[exp][i]['meds'][x]['err_s']   
                 error_h = exps[exp][i]['meds'][x]['err_h']   
                 med_h = exps[exp][i]['meds'][x]['med_h']
                 
                 if(error_s < 7.5 and  error_h < med_h):
                     exps[exp][i]['meds'][x].update({"DENTRO":"SI"})
                     si_cta += 1
                     
                 else:
                     exps[exp][i]['meds'][x].update({"DENTRO":"NO"})
           
           
           #Determine the presence of the drone inside a volume cell
           for x in list(exps[exp][i]['meds_drone'].keys()):
                 error_s = exps[exp][i]['meds_drone'][x]['err_s']   
                 error_h = exps[exp][i]['meds_drone'][x]['err_h']   
                 med_h = exps[exp][i]['meds_drone'][x]['med_h']
                 
                 if(error_s < 7.5 and  error_h < med_h):
                     exps[exp][i]['meds_drone'][x].update({"DENTRO":"SI"})
                     dr_cta += 1
                     
                 else:
                     exps[exp][i]['meds_drone'][x].update({"DENTRO":"NO"})
           
           
           exps[exp][i].update({"yes":si_cta})
           exps[exp][i].update({"yes_dr":dr_cta})          
           
           plt.grid()
           plt.plot()
           plt.xticks([0,40,80,120,160,200,240,280])
           plt.yticks([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140])
           plt.title("Main cells for " + exps[exp][i]["time"])
           title = exps[exp][i]["time"].replace(':','')
           #plt.savefig(PATH_PLOTS[exp-1] +'//'+title+'.png')
          
                
        except:
            pass
        
#%%



OFFSET = 38.5                                              # Offset value for the Azimuth correction 

beam_width_deg = 1.8                                       # Beam width of the transmitting antenna [??]
beam_width = np.deg2rad(1.8)                               # Beam width of the transmitting antenna [rad]
pulse_width = 0.1e-6                                       # Pulse width of the transmitted pulse [s]
c = 3e8                                                    # Speed of light
sigma_r = 0.35*pulse_width*c/2                             # Standard deviation of the pulse in range
sigma_xy = beam_width_deg/2.36                             # Standard deviation of the beam (Azimuth and Elevation)
r_esf = 0.1765                                             # Sphere radius [m]
rcs = round(np.pi*r_esf**2, 2)                             # Radar Cross Section (Optic Region)
freq_oper = 9.345e9                                        # Radar operative frequency
w_length = round(c/freq_oper, 3)                           # Wavelength of the rad
antenna_gain = 10**(38.5/10)                               # Antenna Gain [dBi]
k_m = 0.93                                                 # Atmospheric refractive index 
pulse_width = 0.1e-6                                       # Pulse width of the transmitted pulses [s]
t_power = 1.91                                             # Transmitted Power [W]
WG_len = 7.87                                              # Waveguide length (2.4m) [ft]


#LOSSES
alfa = 1.4e-2*0.2                                          # Atmospheric attenuation
L_circu = 0.5                                              # Circulator losses
L_rot_joint = 0.6                                          # Rotary Joint losses
L_wg = 0.1*WG_len                                          # Waveguide losses
L_adap = 0.5                                               # SMA-WR90 adapter losses
Glna = 10**(83.5/10) 

#L_total = 10**( (alfa + L_circu + L_rot_joint + L_wg + L_adap)/10)     # Total losses
L_total = 10**( (2*alfa + 2*L_circu + 2*L_wg + 4*L_adap)/10)            # Total losses
L_total_db = 2*alfa + 2*L_circu + 2*L_wg + 4*L_adap

#RHI PLOT
a = 6374
ae = 4/3*a

def get_Constant(Pt, G, Glna, lamb, rcs, L):
    
    
    
    """
    Obtain the most important columns in the dataset from a certain hdf5 file for an experiment.
    
    Arguments:
        
             PATH: path of a certain hdf5 file (string)
        
    
    Returns:
         
        a list with the most import columns from the experiment
        
        a[0]: datetime series
        a[1]: height of the drone series 
        a[2]: distance from the radar to the drone series 
        a[3]: height of the sphere series 
        a[4]: distance from the radar to the sphere series 
        a[5]: distance from the radar to the drone within the x component series 
        a[6]: distance from the radar to the drone within the y component series 
        a[7]: distance from the drone to the sphere within its relative x component series 
        a[8]: distance from the drone to the sphere within its relative y component series 
        
    """
        
    C_exp = (Pt*(G**2)*Glna*(lamb**2)*rcs)/(((4*np.pi)**3)*(L))
    return C_exp

#Iterate over each experiment
for exp in range(1,7):
     
    l_28 = {} 
    l_29 = {} 
    l_30 = {}
    
    
    time_pos = getdataset_Exp(PATHS_TAB[exp-1])[0]
    roll_p = getdataset_Exp(PATHS_TAB[exp-1])[9]
    pitch_p = getdataset_Exp(PATHS_TAB[exp-1])[10]
    esf_h_arr = getdataset_Exp(PATHS_TAB[exp-1])[3]
    esf_s_arr = getdataset_Exp(PATHS_TAB[exp-1])[4]
    
    
    x_drone = getdataset_Exp(PATHS_TAB[exp-1])[5]
    y_drone = getdataset_Exp(PATHS_TAB[exp-1])[6]
    e_sph = getdataset_Exp(PATHS_TAB[exp-1])[7]
    f_sph = getdataset_Exp(PATHS_TAB[exp-1])[8]
    drone_h_arr = getdataset_Exp(PATHS_TAB[exp-1])[1]
    drone_s_arr = getdataset_Exp(PATHS_TAB[exp-1])[2]
    
    
    #Columns for building the table
    date = []
    c_initial = []
    c_initial_db = []
    time_W = []
    power = []
    power_db = []
    azimuth = []
    file = []
    range_r = []
    ro_max  =[]
    wr = []
    wb = []
    c_after = []
    c_after_db = []
    c_after_wb = []
    c_after_wb_db = []
    c_after_wb_only = []
    c_after_wb_only_db = []
    wb_x = []
    wb_y = []
    
    theta_x_bar = []
    theta_x = []
    theta_y_bar = []
    theta_y = []
    
    roll_l = []
    pitch_l = []
    
    exp_const_l = []
    exp_const_l_db = []

    #Iterate over each file/sample
    for i in list(exps[exp].keys()):
        try:
             
            for j in list(exps[exp][i]['meds'].keys()):
                 if(exps[exp][i]['meds'][j]['DENTRO'] == "SI"):
                     #idx_hit = list(exps[exp][i]['meds'].keys()).index(j)
                     
                     #Radar variables
                     perf_max = exps[exp][i]['esfera'][j][1]
                     range_max = exps[exp][i]['esfera'][j][2]*15
                    
                     
                     
                     #Experimental constant
                     
                     exp_const = get_Constant(t_power, antenna_gain, Glna, w_length, rcs, L_total)
                     exp_const_db = 10*np.log10(exp_const)
                     
                     #Sphere time
                     
                     #Timestamp from the h5 file as a base time
                     date2, time2 = exps[exp][i]['time'].split("  ")
                     hh,mm,ss = time2.split(":")
                     time_base = datetime.datetime(2022, 5, dias[exp-1], int(hh), int(mm), int(ss))
                     
                     
                     perf_zero = exps[exp][i]['profiles_H'][0][0]
                     dif_ang_esf = round((perf_zero-perf_max)/10,1)
                     dif_secs_esf = math.floor(dif_ang_esf)
                     dif_decs_esf = int(round(math.modf(dif_ang_esf)[0],1)*10)
                     time_sphere = time_base + datetime.timedelta(seconds = dif_secs_esf)
                     idx_sphere = time_pos.index(time_sphere)
                     idx_sphere_f = int(idx_sphere + 10*dif_decs_esf/2)
                     #print(exp, i, dif_secs_esf)
                     
                     h = esf_h_arr[idx_sphere_f] - 2.9
                     l = esf_s_arr[idx_sphere_f]
                     r = np.sqrt(h**2 + l**2)
                     #print(exp, j, range_max, perf_max, r)
                     
                     h_drone = drone_h_arr[idx_sphere_f] - 2.9
                     l_drone = drone_s_arr[idx_sphere_f] 
                     
                     
                     # RHI Plot
                     
                     n_ele = np.array(exps[exp][i]['elevation'])
                     n_ran = np.array(exps[exp][i]['range'])
                     n_azi = np.array(exps[exp][i]['azimuth'])
                     
                     power = exps[exp][i]['powerH']
                     power_list = power.tolist()
                     corr_power = []
                     
                     rows = power.shape[0]
                     
                     for row in range(0,len(power_list)-(d_el[exp-1] + 1)):
                         corr_power.append(power[row + d_el[exp-1]])
                     
                     for k in range(row +1, rows):
                         corr_power.append(-55*np.ones(32))
                     
                     
                     power_f = np.array(corr_power)
                     
                     r2, el_rad2 = np.meshgrid(n_ran, n_ele/180*np.pi)
                     r21 = r2[:,-33:]
                     el_rad21 = el_rad2[:,-33:]
                     ads = np.multiply(r21,np.sin(el_rad21))
                     ads2 = np.multiply(r21,np.cos(el_rad21))
                     
                      
                     y = (r21**2 + ae**2 + 2*ads*ae)**0.5 - ae
                     x = ae*np.arcsin(np.divide(ads2,ae+y))
                     
                     fig, ax = plt.subplots(1,1,figsize=(15,15))
                     
                     #mesh = ax.pcolormesh(x,y,power_f,shading='flat', vmin=-45, vmax=-25, edgecolors='k', linewidths=1)
                     ax.pcolormesh(x,y,power_f,shading='flat', vmin=-60, vmax=0, edgecolors='k', linewidths=1)
                     
                     axq = np.linspace(0, 0.5, 50)
                     axt = 0.04*np.ones(50)
                     
                     
                     ax.plot(l/1000,h/1000,'o-',color="red",linewidth=24)
                     ax.plot(l_drone/1000,h_drone/1000,'o-',color="red",linewidth=15)
                     
                     ax.set(ylim=(0,0.15))
                     ax.set(xlim=(0,0.5))
                     plt.title("RHI para " + str(exp) + " " +  i)
                     plt.xlabel("Rango [Km]")
                     plt.ylabel("Altura [m]")
                     plt.colorbar(ax.pcolormesh(x,y,power_f,shading='flat', vmin=-60, vmax=0, edgecolors='k', linewidths=1))
                     plt.jet()
                     title = i
                     title = title.replace("  ", "x")
                     title = title.replace("-","")
                     title = title.replace(":","")
                     plt.savefig(PATH_RHI[exp-1]+'//'+i[-11:-7]+'//'+title+'.png')
                    
                     plt.legend(['Dos','Tres'])
                     # New RWF BWF
                     
                     #maxvalInRows = np.amax(power_f, axis=1)
                     maxRow = np.argmax(power_f[14], axis = 0)
                     #maxCol = np.argmax(power_f[14], axis = 0)
                     
                     
                     r_o = 217.5    
                    
                     #r_power = power_f[maxRow, 14]
                     r_power = 2
                     #Calculate the radar calibration constant...
                     
                     C_initial = r_power*(r**4)
     
                     Wr = np.exp(-((r-r_o)**2)/(2*sigma_r**2))
                     
                     C_after = (r_power*(r**4))/(Wr)
                    
                     ##
                     
                     y_d = y_drone[idx_sphere_f]
                     x_d = x_drone[idx_sphere_f]
                     e_esf = e_sph[idx_sphere_f]
                     f_esf = f_sph[idx_sphere_f]
                     roll = roll_p[idx_sphere_f]
                     pitch = pitch_p[idx_sphere_f]
                     
                     
                     theta_X_bar = float(i[-11:-7])
                     theta_Y_bar = exps[exp][i]['elevation'][maxRow]
                     
                     gamma = np.rad2deg(np.arctan((y_d)/(x_d)))
                     theta =  OFFSET - gamma
                     alfa =  np.rad2deg(np.arctan(f_esf/l))
                     
                     theta_X = theta + alfa
                     theta_Y = np.rad2deg(np.arctan((h)/l))
                     
                     Wb = np.exp(-((theta_X-theta_X_bar)**2)/(2*sigma_xy**2)     -((theta_Y-theta_Y_bar)**2)/(2*sigma_xy**2))
                     Wb_x = np.exp(-((theta_X-theta_X_bar)**2)/(2*sigma_xy**2))
                     Wb_y = np.exp(-((theta_Y-theta_Y_bar)**2)/(2*sigma_xy**2))
                     
                  
                   
                     C_after_wb = (r_power*(r**4))/(Wr*Wb)
                     
                     C_after_wb_only = (r_power*(r**4))/(Wb)
                     
                     #Adding each processed variable from a sample to a list 
                     date.append(exps[exp][i]['time'])
                     range_r.append(r)
                     wr.append(Wr)
                     wb.append(Wb)
                     wb_x.append(Wb_x)
                     wb_y.append(Wb_y)
                     theta_y_bar.append(theta_Y_bar)
                     theta_y.append(theta_Y)
                     theta_x_bar.append(theta_X_bar)
                     theta_x.append(theta_X)
                     
                     ro_max.append(r_o)
                     power.append(10**(r_power/10))
                     power_db.append(r_power)
                     
                     c_initial.append(C_initial)
                     c_initial_db.append(10*np.log10(C_initial))
                     
                     #Az
                     azimuth.append(float(i[-11:-7]))
                     c_after.append(C_after)
                     c_after_db.append(10*np.log10(C_after))
                     
                     c_after_wb.append(C_after_wb)
                     c_after_wb_db.append(10*np.log10(C_after_wb))
                     
                     c_after_wb_only.append(C_after_wb_only)
                     c_after_wb_only_db.append(10*np.log10(C_after_wb_only))
                     
                     file.append(i)
                     
                     roll_l.append(roll)
                     pitch_l.append(pitch)
                    
                     exp_const_l.append(exp_const)
                     exp_const_l_db.append(exp_const_db)
             
        except:
             
            pass
    
    
    
    data = [date, file, roll_l, pitch_l, azimuth, ro_max, range_r, power, power_db, wr, wb, 
            c_initial, c_initial_db, c_after, c_after_db,c_after_wb_only, c_after_wb_only_db,
            c_after_wb, c_after_wb_db, theta_x_bar, theta_x, theta_y_bar, theta_y, wb_x, wb_y,
            exp_const_l, exp_const_l_db]      
   
    df = pd.DataFrame(data)
    df = df.transpose()

    
    df.columns = ['Datetime','Filename','Roll','Pitch','Azimuth', 'r_o','range','R Power [W]',
                  'R Power [dB]','RWF', 'BWF','C_initial', 'C_initial [dB]','C_after Wr', 
                  'C_after Wr [dB]','C_after Wb Only','C_after Wb Only [dB]','C_after Wb', 
                  'C_after Wb [dB]','Theta X bar', 'Theta X','Theta Y bar','Theta Y','Wb x', 
                  'Wb y','Exp Constant', 'Exp Constant [dB]']
    df.to_excel(r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Post_processing\Tables_after_rhi'+'\\'+'Table_exp'+str(exp)+'.xlsx', sheet_name='tabla')
    
    #%%
    
test = exps[5]['SOPHY_20220509_085411_A28.2_S.hdf5']['powerH']

test_l = test.tolist()
test_corr = []

b = test.shape[0]
print(b)

for row in range(0,len(test)-9,1):
    test_corr.append(test[row + 8])
    
for i in range(row + 1, b):
    test_corr.append(-55*np.ones(32))
    
    
new_arr = np.array(test_corr)

    