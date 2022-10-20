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
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


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


SAVE_PATH_1 = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\Plots_power\Plots_exp_1'
SAVE_PATH_2 = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\Plots_power\Plots_exp_2'
SAVE_PATH_3 = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\Plots_power\Plots_exp_3'
SAVE_PATH_4 = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\Plots_power\Plots_exp_4'
SAVE_PATH_5 = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\Plots_power\Plots_exp_5'
SAVE_PATH_6 = r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\Plots_power\Plots_exp_6'


PATHS = [PATH_1, PATH_2, PATH_3, PATH_4, PATH_5, PATH_6]

PATHS_TAB = [POS_1, POS_2, POS_3, POS_4, POS_5, POS_6]


PATH_ALL = r'C:\Users\GIBS\Documents\Experimentos\Exp5_Final\Drone\plots_drone' 

PATH_PLOTS = [SAVE_PATH_1, SAVE_PATH_2, SAVE_PATH_3, SAVE_PATH_4, SAVE_PATH_5, SAVE_PATH_6]

dias = [7,8,8,8,9,9]  # Day when the experiment was realized
#d_el = [5,5,7,7,7,7]  # Delay in EL associated to certain experiment
d_el = [4,4,6,6,6,6]  # Delay in EL associated to certain experiment
OFF = 1 # 15m
exps = {}


count = 1


#################
INT_MAX = 10000
 
# Given three collinear points p, q, r, 
# the function checks if point q lies
# on line segment 'pr'
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
     
    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True
         
    return False
 
# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p:tuple, q:tuple, r:tuple) -> int:
     
    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
           ((q[0] - p[0]) *
            (r[1] - q[1])))
            
    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock
 
def doIntersect(p1, q1, p2, q2):
     
    # Find the four orientations needed for 
    # general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if (o1 != o2) and (o3 != o4):
        return True
     
    # Special Cases
    # p1, q1 and p2 are collinear and
    # p2 lies on segment p1q1
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True
 
    # p1, q1 and p2 are collinear and
    # q2 lies on segment p1q1
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True
 
    # p2, q2 and p1 are collinear and
    # p1 lies on segment p2q2
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True
 
    # p2, q2 and q1 are collinear and
    # q1 lies on segment p2q2
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True
 
    return False
 
# Returns true if the point p lies 
# inside the polygon[] with n vertices
def is_inside_polygon(points:list, p:tuple) -> bool:
     
    n = len(points)
     
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
         
    # Create a point for line segment
    # from p to infinite
    extreme = (INT_MAX, p[1])
     
    # To count number of points in polygon
      # whose y-coordinate is equal to
      # y-coordinate of the point
    decrease = 0
    count = i = 0
     
    while True:
        next = (i + 1) % n
         
        if(points[i][1] == p[1]):
            decrease += 1
         
        # Check if the line segment from 'p' to 
        # 'extreme' intersects with the line 
        # segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(points[i],
                        points[next],
                        p, extreme)):
                             
            # If the point 'p' is collinear with line 
            # segment 'i-next', then check if it lies 
            # on segment. If it lies, return true, otherwise false
            if orientation(points[i], p,
                           points[next]) == 0:
                return onSegment(points[i], p,
                                 points[next])
                                  
            count += 1
             
        i = next
         
        if (i == 0):
            break
             
    # Reduce the count by decrease amount
      # as these points would have been added twice
    count -= decrease
     
    # Return true if count is odd, false otherwise
    return (count % 2 == 1)

###############################3

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
        
    return [time_list, df.z_d, df.l_d, df.z_e, df.l_e, df.x_d, df.y_d, df.E_l, df.F_l]


def get_coords(ang, rangearr):    
    
    """
    Get the s-h cartesian coordinates of center of the resolution cell using the elevation 
    profile and the range cell.
    
    Arguments:
        
             ang: profile of the pedestal [°] (float)
        rangearr: range cell of the radar as the index from the range array (integer)
        
        
    Returns:
         
        a list with the coordinates of the estimated resolution cell
        
        a[0]: lower left corner of the rectangle
        a[1]: height of lower side (horizontal) of the resolution cell (0 - height) 
        a[2]: height of the resolution cell volume (upper height - lower height)
        a[3]: height of the center of the resolution cell
        
    """
    
    #center = 15*rangearr
    #inner_center = 15*rangearr - 7.5
    #outer_center = 15*rangearr + 7.5
    
    center = 15*rangearr + 7.5
    inner_center = 15*rangearr  
    outer_center = 15*rangearr + 15
    
    
    upper_left_x = inner_center*(np.cos(np.deg2rad(ang+0.9)))
    upper_left_y = inner_center*(np.sin(np.deg2rad(ang+0.9)))
    
    lower_left_x = inner_center*(np.cos(np.deg2rad(ang-0.9)))
    lower_left_y = inner_center*(np.sin(np.deg2rad(ang-0.9)))
    
    upper_right_x = outer_center*(np.cos(np.deg2rad(ang+0.9)))
    upper_right_y = outer_center*(np.sin(np.deg2rad(ang+0.9)))
    
    lower_right_x = outer_center*(np.cos(np.deg2rad(ang-0.9)))
    lower_right_y = outer_center*(np.sin(np.deg2rad(ang-0.9)))
    
    poly = np.array([upper_left_x,upper_left_y, lower_left_x,lower_left_y, lower_right_x,
                   lower_right_y, upper_right_x, upper_right_y]).reshape(4,2)
    
    centerx = center*(np.deg2rad(np.cos(ang)))
    centery = center*(np.deg2rad(np.sin(ang)))
        
    a = [centerx, centery, poly]

    return a
'''
def get_coords(ang, rangearr):    
    
    first_range = (15*rangearr)*np.cos(np.deg2rad(ang))
    center = first_range
    alt_centro = (15*rangearr)*np.sin(np.deg2rad(ang))
    alt_sup = (15*rangearr)*np.sin(np.deg2rad(ang+0.9))
    alt_inf = (15*rangearr)*np.sin(np.deg2rad(ang-0.9))
    
    dif_alt = alt_sup - alt_inf
        
    a = [center - 7.5, alt_inf, dif_alt, alt_centro]

    return a
'''
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
           
           
           #Pr*(r**4)
           for j in exps[exp][i]['esfera'].keys():
               
               h = esf_h_arr[idx_sphere_f] - 2.9
               s = esf_s_arr[idx_sphere_f]
               
               r = np.sqrt(h**2 + s**2)
               
               prrangef = exps[exp][i]['esfera'][j][4]*(r**4)
               prrangeflog = 10*np.log10(prrangef)
               
               exps[exp][i]['esfera'][j].append(r)
               exps[exp][i]['esfera'][j].append(prrangef)
               exps[exp][i]['esfera'][j].append(prrangeflog)
           
           
           for j in exps[exp][i]['drone'].keys():
               
               corr_perf = exps[exp][i]['profiles_H'][ exps[exp][i]['drone'][j][3]-d_el[exp-1]][0]
               range_perf = exps[exp][i]['drone'][j][2]
               #esfera_c = get_coords(j[1],k)
               drone_c = get_coords(corr_perf,range_perf)
               ax.add_patch(Polygon(drone_c[2], edgecolor = 'black', facecolor = 'lightblue', fill=True))
               plt.plot(drone_c[0],drone_c[1],'xw', markersize = 12)
               '''
               power =  exps[exp][i]['powerH'].shape[0]
               potentials_d.update({j:{"err_s":np.abs(drone_c[0] + 7.5 -drone_s_arr[idx_drone]),
                                     "err_h":np.abs(drone_c[3] - (drone_h_arr[idx_drone] -2.9)),
                                     "med_h":drone_c[3]-drone_c[1]}})    
               '''
               
           exps[exp][i].update({"meds_drone":potentials_d})
           
           if( len(exps[exp][i]['drone']) != 0): 
               max_power_drone_idx = rpower_drone.index(max(rpower_drone)) 
               max_drone_refer = list(exps[exp][i]['drone'].keys())[max_power_drone_idx]
               max_range =  exps[exp][i]['drone'][max_drone_refer][2]
               max_perf = exps[exp][i]['drone'][max_drone_refer][1]
               drone_max = get_coords(max_perf, max_range)
               ax.add_patch(Polygon(drone_max[2], edgecolor = 'black', facecolor = 'yellow', fill=True))
               plt.plot(drone_max[0], drone_max[1],'xw', markersize = 12)
              
              
               
           for j in exps[exp][i]['esfera'].keys():
               
              #
              
               powerrfo = exps[exp][i]['esfera'][j][-1]
              
    
               if (powerrfo < 45.0): 
                  color = "aquamarine"
                  
               elif(45.0 < powerrfo < 55.0): 
                  color = "lime"
                  
               elif(55.0 < powerrfo < 65.0): 
                  color = "yellow" 
                  
               elif(65.0 < powerrfo < 75.0): 
                  color = "orange"    
                 
               elif(75.0 < powerrfo < 85.0): 
                  color = "firebrick"       
            
                
               corr_perf = exps[exp][i]['profiles_H'][ exps[exp][i]['esfera'][j][3]-d_el[exp-1]][0]
               range_perf = exps[exp][i]['esfera'][j][2]
               #esfera_c = get_coords(j[1],k)
               esfera_c = get_coords(corr_perf,range_perf)
               ax.add_patch(Polygon(esfera_c[2],edgecolor = 'black', facecolor = color, fill=True))
               plt.plot(esfera_c[0], esfera_c[1],'xw', markersize = 12)
              
                
               polygon1 = [ tuple(esfera_c[2][0]), tuple(esfera_c[2][1]), tuple(esfera_c[2][2]), tuple(esfera_c[2][3]) ]
                     
               p = (esf_s_arr[idx_sphere_f], esf_h_arr[idx_sphere_f]-2.9)
                
               if (is_inside_polygon(points = polygon1, p = p)):
                   potentials.update({j:"SI"})
               else:
                   potentials.update({j:"NO"})
                   
               
               
               
               
               
               '''
               power =  exps[exp][i]['powerH'].shape[0]
                   
               potentials.update({j:{"err_s":np.abs(esfera_c[0] + 7.5 -esf_s_arr[idx_sphere_f]),
                                     "err_h":np.abs(esfera_c[3] - (esf_h_arr[idx_sphere_f] -2.9)),
                                     "med_h":esfera_c[3]-esfera_c[1]}})
               
           exps[exp][i].update({"meds":potentials})
           
               '''
           
           exps[exp][i].update({"meds":potentials})
           ''' 
           if( len(exps[exp][i]['esfera']) != 0): 
               max_power_sph_idx = rpower_sphere.index(max(rpower_sphere)) 
               max_sph_refer = list(exps[exp][i]['esfera'].keys())[max_power_sph_idx]
               max_range =  exps[exp][i]['esfera'][max_sph_refer][2]
               max_perf = exps[exp][i]['esfera'][max_sph_refer][1]
              
               
               esfera_max = get_coords(max_perf, max_range)
               ax.add_patch(Polygon(esfera_max[2],edgecolor = 'black', facecolor = 'green', fill=True))
              
               plt.plot(esfera_max[0], esfera_max[1],'xw', markersize = 12)
           '''
               
           si_cta = 0
           dr_cta = 0
           
           '''
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
           '''
           
           scale_img = mpimg.imread('scale.png')
           imagebox = OffsetImage(scale_img, zoom=0.5)
           ab = AnnotationBbox(imagebox, (60, 30))
           ax.add_artist(ab)
           plt.xlabel("S [m]")
           plt.ylabel("H [m]")
           plt.grid()
           plt.draw()
           plt.plot()
           plt.xticks([0,40,80,120,160,200,240,280])
           plt.yticks([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140])
           plt.title("Echoes for " + exps[exp][i]["time"])
           title = exps[exp][i]["time"].replace(':','')
           plt.savefig(PATH_PLOTS[exp-1] +'//'+i[-11:-7]+'//'+title+'.png')
               
        
                
        except:
            pass
        
#%%



OFFSET = 38.5                                              # Offset value for the Azimuth correction 

beam_width_deg = 1.8                                       # Beam width of the transmitting antenna [°]
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
    esf_h_arr = getdataset_Exp(PATHS_TAB[exp-1])[3]
    esf_s_arr = getdataset_Exp(PATHS_TAB[exp-1])[4]
    x_drone = getdataset_Exp(PATHS_TAB[exp-1])[5]
    y_drone = getdataset_Exp(PATHS_TAB[exp-1])[6]
    e_sph = getdataset_Exp(PATHS_TAB[exp-1])[7]
    f_sph = getdataset_Exp(PATHS_TAB[exp-1])[8]
    
    
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
    
    theta_y = []
    theta_y_bar = []
    theta_x = []
    theta_x_bar = []
    
    exp_const_l = []
    exp_const_l_db = []
    

    #Iterate over each file/sample
    for i in list(exps[exp].keys()):
        try:
             
            for j in list(exps[exp][i]['meds'].keys()):
                 if(exps[exp][i]['meds'][j] == "SI"):
                     #idx_hit = list(exps[exp][i]['meds'].keys()).index(j)
                     
                     #Radar variables
                     perf_max = exps[exp][i]['esfera'][j][1]
                     range_max = exps[exp][i]['esfera'][j][2]*15
                     r_power = exps[exp][i]['esfera'][j][4]
                     
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
                     print(exp, j, range_max, perf_max, r)
                     
                     #Calculate the radar calibration constant...
                     
                     C_initial = r_power*(r**4)
     
                     Wr = np.exp(-((r-(range_max+7.5))**2)/(2*sigma_r**2))
                     
                     
                     
                     C_after = (r_power*(r**4))/(Wr)
                    
                     ##
                     
                     y_d = y_drone[idx_sphere_f]
                     x_d = x_drone[idx_sphere_f]
                     e_esf = e_sph[idx_sphere_f]
                     f_esf = f_sph[idx_sphere_f]
                     
                     theta_X_bar = float(i[-11:-7])
                     theta_Y_bar = perf_max
                     
                     gamma = np.rad2deg(np.arctan((y_d)/(x_d)))
                     theta =  OFFSET - gamma
                     alfa =  np.rad2deg(np.arctan(f_esf/l))
                     
                     theta_X = theta + alfa
                     theta_Y = np.rad2deg(np.arctan((h)/l))
                     
                     Wb = np.exp(-((theta_X-theta_X_bar)**2)/(2*sigma_xy**2) -((theta_Y-theta_Y_bar)**2)/(2*sigma_xy**2))
                   
                     C_after_wb = (r_power*(r**4))/(Wr*Wb)
                     
                
                     #Adding each processed variable from a sample to a list 
                     date.append(exps[exp][i]['time'])
                     range_r.append(r)
                     wr.append(Wr)
                     wb.append(Wb)
                     ro_max.append(range_max + 7.5)
                     power.append(r_power)
                     power_db.append(10*np.log10(r_power))
                     
                     c_initial.append(C_initial)
                     c_initial_db.append(10*np.log10(C_initial))
                     
                     #Az
                     azimuth.append(float(i[-11:-7]))
                     c_after.append(C_after)
                     c_after_db.append(10*np.log10(C_after))
                     
                     c_after_wb.append(C_after_wb)
                     c_after_wb_db.append(10*np.log10(C_after_wb))
                     
                     
                     ###
                     theta_x_bar.append(theta_X_bar)
                     theta_x.append(theta_X)
                     
                     theta_y_bar.append(theta_Y_bar)
                     theta_y.append(theta_Y)
                     ###
                     exp_const_l.append(exp_const)
                     exp_const_l_db.append(exp_const_db)
                     
                     file.append(i)
                    
             
        except:
             
            pass
    
    
    data = [date, file,azimuth, ro_max, range_r, power, power_db, wr, wb, 
            c_initial, c_initial_db, c_after, c_after_db, c_after_wb, 
            c_after_wb_db, theta_x_bar, theta_x, theta_y_bar, theta_y,
            exp_const_l, exp_const_l_db]  
   
    df = pd.DataFrame(data)
    df = df.transpose()

    
    df.columns = ['Datetime','Filename','Azimuth', 'r_o','range','R Power [W]','R Power [dB]','RWF', 'BWF',
                  'C_initial', 'C_initial [dB]','C_after', 'C_after [dB]','C_after_wb', 'C_after_wb [dB]',
                  'Theta x bar', 'Theta x', 'Theta y bar','Theta y', 'Exp Constant', 'Exp Costant [dB]']
    df.to_excel(r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Post_processing\Tables_after_wr_wb_new'+'\\'+'Table_exp'+str(exp)+'.xlsx', sheet_name='tabla')
#%%

PATHPLOT =r'C:\Users\GIBS\Documents\Experimentos\Experimento2\Plots_ch0'
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
l_power = []
a = 6374
ae = 4/3*a

for i in dict_3deg_time.keys():

    file = h5.File(PATH+"\\"+i, 'r')
    
    n_ele = file['Metadata']['elevation']
    n_ran = file['Metadata']['range']
    n_azi = file['Metadata']['azimuth']
    
    power = file['Data']['data_param']['channel00']
    
    Data_Arr = 10*np.log10(power[:,-33:])
    #Data_Arr[0:51,:10] = -55.0
    #Data_Arr[Data_Arr > -17] = -55.0
    l_power.append(Data_Arr)
    n_ran2 = np.array(n_ran)
    n_ele2 = np.array(n_ele)
    
    
    r2, el_rad2 = np.meshgrid(n_ran2, n_ele2/180*np.pi)
    
    r21 = r2[:,-33:]
    el_rad21 = el_rad2[:,-33:]
    ads = np.multiply(r21,np.sin(el_rad21))
    ads2 = np.multiply(r21,np.cos(el_rad21))
    
    y = (r21**2 + ae**2 + 2*ads*ae)**0.5 - ae
    x = ae*np.arcsin(np.divide(ads2,ae+y))
    
    
    fig, ax = plt.subplots(1,1,figsize=(15,15))
    ax.pcolormesh(x,y,Data_Arr,shading='flat', vmin=-45, vmax=-25, edgecolors='k', linewidths=1)
    ax.set(ylim=(0,0.1))
    ax.set(xlim=(0,0.5))
    plt.title("RHI para " + dict_3deg_time[i])
    plt.xlabel("Rango [Km]")
    plt.ylabel("Altura [m]")
    title = dict_3deg_time[i]
    title = title.replace("  ", "x")
    title = title.replace("-","")
    title = title.replace(":","")
    plt.savefig(PATHPLOT+"\\"+title+".png")
    
    
#%%


INT_MAX = 10000
 
# Given three collinear points p, q, r, 
# the function checks if point q lies
# on line segment 'pr'
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
     
    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True
         
    return False
 
# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p:tuple, q:tuple, r:tuple) -> int:
     
    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
           ((q[0] - p[0]) *
            (r[1] - q[1])))
            
    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock
 
def doIntersect(p1, q1, p2, q2):
     
    # Find the four orientations needed for 
    # general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if (o1 != o2) and (o3 != o4):
        return True
     
    # Special Cases
    # p1, q1 and p2 are collinear and
    # p2 lies on segment p1q1
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True
 
    # p1, q1 and p2 are collinear and
    # q2 lies on segment p1q1
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True
 
    # p2, q2 and p1 are collinear and
    # p1 lies on segment p2q2
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True
 
    # p2, q2 and q1 are collinear and
    # q1 lies on segment p2q2
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True
 
    return False
 
# Returns true if the point p lies 
# inside the polygon[] with n vertices
def is_inside_polygon(points:list, p:tuple) -> bool:
     
    n = len(points)
     
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
         
    # Create a point for line segment
    # from p to infinite
    extreme = (INT_MAX, p[1])
     
    # To count number of points in polygon
      # whose y-coordinate is equal to
      # y-coordinate of the point
    decrease = 0
    count = i = 0
     
    while True:
        next = (i + 1) % n
         
        if(points[i][1] == p[1]):
            decrease += 1
         
        # Check if the line segment from 'p' to 
        # 'extreme' intersects with the line 
        # segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(points[i],
                        points[next],
                        p, extreme)):
                             
            # If the point 'p' is collinear with line 
            # segment 'i-next', then check if it lies 
            # on segment. If it lies, return true, otherwise false
            if orientation(points[i], p,
                           points[next]) == 0:
                return onSegment(points[i], p,
                                 points[next])
                                  
            count += 1
             
        i = next
         
        if (i == 0):
            break
             
    # Reduce the count by decrease amount
      # as these points would have been added twice
    count -= decrease
     
    # Return true if count is odd, false otherwise
    return (count % 2 == 1)


polyre = get_coords(30, 14)



polygon1 = [ tuple(polyre[2][0]), tuple(polyre[2][1]), tuple(polyre[2][2]), tuple(polyre[2][3]) ]
     
p = (186, 108)

if (is_inside_polygon(points = polygon1, p = p)):
   print ('Yes')
else:
   print ('No')

#%%
expe = list(exps[2].keys())  
offset = [d_el[1]]*len(list(exps[2].keys()))

data = [expe, offset]

df = pd.DataFrame(data)
df = df.transpose()

    
df.columns = ['Experiment','Offset in El (Samples)']


df.to_excel(r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing'+'\\'+'Exp2_delay.xlsx', sheet_name='tabla')
#%%

df = pd.ExcelFile(r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing'+'\\'+'Exp2_delay.xlsx').parse('tabla')

delay_exp2 = df['Offset_corr']

####
time_pos = getdataset_Exp(PATHS_TAB[1])[0]
drone_h_arr = getdataset_Exp(PATHS_TAB[1])[1]
drone_s_arr = getdataset_Exp(PATHS_TAB[1])[2]

esf_h_arr = getdataset_Exp(PATHS_TAB[1])[3]
esf_s_arr = getdataset_Exp(PATHS_TAB[1])[4]


#count_xp = 0
#Iterate over each file/sample
for i in range(len(list(exps[2].keys()))):
    profiles_H = []
    profiles_H_fxd = []
    powerH = exps[2][list(exps[2].keys())[i]]['powerH']
    
    #Iterate over each row from the power dataset of the sample 
    for j in range(powerH.shape[0]):
        if (len(np.where(powerH[j] > -40.0)[0])!= 0 and powerH[j][10]<-25.):
            
            profiles_H.append([exps[2][list(exps[2].keys())[i]]['elevation'][j], list(np.where(powerH[j] > -40.0)[0])])
            profiles_H_fxd.append([exps[2][list(exps[2].keys())[i]]['elevation'][j-delay_exp2[i]], list(np.where(powerH[j] > -40.0)[0])])
            
        else:
            profiles_H.append([exps[2][list(exps[2].keys())[i]]['elevation'][j], []])
            profiles_H_fxd.append([exps[2][list(exps[2].keys())[i]]['elevation'][j-delay_exp2[i]], []])
    
    exps[2][list(exps[2].keys())[i]].update({"profiles_H":profiles_H})
    exps[2][list(exps[2].keys())[i]].update({"profiles_H_fxd":profiles_H_fxd})
    
    count = 0
    drone = {}
    esfera = {}
    rpower_sphere = []
    rpower_drone = []
    
    
 

    #Separation of power samples associated to the sphere / UAV
    
    for j in range(len(exps[2][list(exps[2].keys())[i]]["profiles_H"])):
        
        if count == 0 :
            if(len(exps[2][list(exps[2].keys())[i]]["profiles_H"][j][1]) != 0):
                for k in exps[2][list(exps[2].keys())[i]]["profiles_H"][j][1]:
                    drone.update({str(exps[2][list(exps[2].keys())[i]]["profiles_H"][j][0]) + " " + str(k) : [exps[2][list(exps[2].keys())[i]]["profiles_H"][j][0],
                                                                                      exps[2][list(exps[2].keys())[i]]["profiles_H"][j-delay_exp2[i]][0], 
                                                                                     k, 
                                                                                     j, 
                                                                                     10**(exps[2][list(exps[2].keys())[i]]['powerH'][j][k]/10)]})

                count = 1
                
        if count == 1:
            if(len(exps[2][list(exps[2].keys())[i]]["profiles_H"][j][1]) != 0):
                for k in exps[2][list(exps[2].keys())[i]]["profiles_H"][j][1]:
                    drone.update({str(exps[2][list(exps[2].keys())[i]]["profiles_H"][j][0]) + " " + str(k) : [exps[2][list(exps[2].keys())[i]]["profiles_H"][j][0],
                                                                                     exps[2][list(exps[2].keys())[i]]["profiles_H"][j-delay_exp2[i]][0], 
                                                                                     k, 
                                                                                     j, 
                                                                                     10**(exps[2][list(exps[2].keys())[i]]["powerH"][j][k]/10)]})
                    
            else:
                count = 2
                
        if count == 2:
            if(len(exps[2][list(exps[2].keys())[i]]["profiles_H"][j][1]) != 0):
                count = 3
        
        if count == 3:
            if(len(exps[2][list(exps[2].keys())[i]]["profiles_H"][j][1]) != 0):
                for k in exps[2][list(exps[2].keys())[i]]["profiles_H"][j][1]:    
                    esfera.update({str(exps[2][list(exps[2].keys())[i]]["profiles_H"][j][0]) + " " + str(k) : [exps[2][list(exps[2].keys())[i]]["profiles_H"][j][0],
                                                                                      exps[2][list(exps[2].keys())[i]]["profiles_H"][j-delay_exp2[i]][0], 
                                                                                      k, 
                                                                                      j, 
                                                                                      10**(exps[2][list(exps[2].keys())[i]]["powerH"][j][k]/10)]})
                    rpower_sphere.append(10**(exps[2][list(exps[2].keys())[i]]["powerH"][j][k]/10))
                    
    exps[2][list(exps[2].keys())[i]].update({"esfera":esfera})
    exps[2][list(exps[2].keys())[i]].update({"drone":drone})
    exps[2][list(exps[2].keys())[i]].update({"rpower_sph":rpower_sphere})
    exps[2][list(exps[2].keys())[i]].update({"rpower_dro":rpower_drone})
    

    #Updating the dict with the maximum received power associated to the sphere 
    if len(rpower_sphere) != 0:    
        exps[2][list(exps[2].keys())[i]].update({"maxpower_sph":max(rpower_sphere)})
    else:                                                                                                 
        exps[2][list(exps[2].keys())[i]].update({"maxpower_sph":0})
        
    for j in list(exps[2][list(exps[2].keys())[i]]['drone'].keys()):
        rpower_drone.append(exps[2][list(exps[2].keys())[i]]['drone'][j][-1])
        
        
    #Plotting each sample with the estimated position of the sphere and the max volume
    fig, ax = plt.subplots(1,1,figsize=(15,15))
    
    #Timestamp from the h5 file
    date2, time2 = exps[2][list(exps[2].keys())[i]]['time'].split("  ")
    hh,mm,ss = time2.split(":")
    time_base = datetime.datetime(2022, 5, dias[1], int(hh), int(mm), int(ss))
    #print(time_base)
     
    #Highest profile (Upper position of the pedestal in RHI mode)
    perf_max = exps[2][list(exps[2].keys())[i]]['profiles_H'][0][0]
    
   
    
    try:
       
       if( len(exps[2][list(exps[2].keys())[i]]['drone']) != 0):
           max_power_drone_idx = rpower_drone.index(max(rpower_drone))
           max_drone_refer = list(exps[2][list(exps[2].keys())[i]]['drone'].keys())[max_power_drone_idx]
           dif_ang_drone = round((perf_max-exps[2][list(exps[2].keys())[i]]['drone'][max_drone_refer][1])/10,1)
           dif_secs_drone = math.floor(dif_ang_drone)
           time_drone = time_base + datetime.timedelta(seconds = dif_secs_drone)
           idx_drone = list(time_pos).index(time_drone)
           #print(idx_drone)
           plt.plot(drone_s_arr[idx_drone],drone_h_arr[idx_drone]-2.9,'ko', linewidth=2, markersize=10)
            
           #Tiempo referencial para la esfera y ploteo
       if( len(exps[2][list(exps[2].keys())[i]]['esfera']) != 0):   
           max_power_sph_idx = rpower_sphere.index(max(rpower_sphere))
           max_sph_refer = list(exps[2][list(exps[2].keys())[i]]['esfera'].keys())[max_power_sph_idx]
           dif_ang_esf = round((perf_max-exps[2][list(exps[2].keys())[i]]['esfera'][max_sph_refer][1])/10,1)
           dif_secs_esf = math.floor(dif_ang_esf)
           dif_decs_esf = int(round(math.modf(dif_ang_esf)[0],1)*10)
           time_sphere = time_base + datetime.timedelta(seconds = dif_secs_esf)
           idx_sphere = list(time_pos).index(time_sphere)
           idx_sphere_f = int(idx_sphere + 10*dif_decs_esf/2)
           
           plt.plot(esf_s_arr[idx_sphere_f],esf_h_arr[idx_sphere_f]-2.9,'ko', linewidth=2, markersize=10)
            
       potentials = {}
       potentials_d = {}
       
       
       #Pr*(r**4)
       for j in exps[2][list(exps[2].keys())[i]]['esfera'].keys():
           
           h = esf_h_arr[idx_sphere_f] - 2.9
           s = esf_s_arr[idx_sphere_f]
           
           r = np.sqrt(h**2 + s**2)
           
           prrangef = exps[2][list(exps[2].keys())[i]]['esfera'][j][4]*(r**4)
           prrangeflog = 10*np.log10(prrangef)
           
           exps[2][list(exps[2].keys())[i]]['esfera'][j].append(r)
           exps[2][list(exps[2].keys())[i]]['esfera'][j].append(prrangef)
           exps[2][list(exps[2].keys())[i]]['esfera'][j].append(prrangeflog)
       

       for j in exps[2][list(exps[2].keys())[i]]['drone'].keys():
           
           corr_perf = exps[2][list(exps[2].keys())[i]]['profiles_H'][ exps[2][list(exps[2].keys())[i]]['drone'][j][3]-delay_exp2[i]][0]
           range_perf = exps[2][list(exps[2].keys())[i]]['drone'][j][2]
           #esfera_c = get_coords(j[1],k)
           drone_c = get_coords(corr_perf,range_perf)
           ax.add_patch(Polygon(drone_c[2], edgecolor = 'black', facecolor = 'lightblue', fill=True))
           plt.plot(drone_c[0],drone_c[1],'xw', markersize = 12)
           '''
           power =  exps[exp][i]['powerH'].shape[0]
           potentials_d.update({j:{"err_s":np.abs(drone_c[0] + 7.5 -drone_s_arr[idx_drone]),
                                 "err_h":np.abs(drone_c[3] - (drone_h_arr[idx_drone] -2.9)),
                                 "med_h":drone_c[3]-drone_c[1]}})    
           '''
           
       #exps[2][list(exps[2].keys())[i]].update({"meds_drone":potentials_d})
       
       if( len(exps[2][list(exps[2].keys())[i]]['drone']) != 0): 
           max_power_drone_idx = rpower_drone.index(max(rpower_drone)) 
           max_drone_refer = list(exps[2][list(exps[2].keys())[i]]['drone'].keys())[max_power_drone_idx]
           max_range =  exps[2][list(exps[2].keys())[i]]['drone'][max_drone_refer][2]
           max_perf = exps[2][list(exps[2].keys())[i]]['drone'][max_drone_refer][1]
           drone_max = get_coords(max_perf, max_range)
           ax.add_patch(Polygon(drone_max[2], edgecolor = 'black', facecolor = 'yellow', fill=True))
           plt.plot(drone_max[0], drone_max[1],'xw', markersize = 12)
          
          
           
       for j in exps[2][list(exps[2].keys())[i]]['esfera'].keys():
           
          #
          
           powerrfo = exps[2][list(exps[2].keys())[i]]['esfera'][j][-1]
          

           if (powerrfo < 45.0): 
              color = "aquamarine"
              
           elif(45.0 < powerrfo < 55.0): 
              color = "lime"
              
           elif(55.0 < powerrfo < 65.0): 
              color = "yellow" 
              
           elif(65.0 < powerrfo < 75.0): 
              color = "orange"    
             
           elif(75.0 < powerrfo < 85.0): 
              color = "firebrick"       
        
            
           corr_perf = exps[2][list(exps[2].keys())[i]]['profiles_H'][ exps[2][list(exps[2].keys())[i]]['esfera'][j][3]-delay_exp2[i]][0]
           range_perf = exps[2][list(exps[2].keys())[i]]['esfera'][j][2]
           #esfera_c = get_coords(j[1],k)
           esfera_c = get_coords(corr_perf,range_perf)
           ax.add_patch(Polygon(esfera_c[2],edgecolor = 'black', facecolor = color, fill=True))
           plt.plot(esfera_c[0], esfera_c[1],'xw', markersize = 12)
          
            
           polygon1 = [ tuple(esfera_c[2][0]), tuple(esfera_c[2][1]), tuple(esfera_c[2][2]), tuple(esfera_c[2][3]) ]
                 
           p = (esf_s_arr[idx_sphere_f], esf_h_arr[idx_sphere_f]-2.9)
            
           if (is_inside_polygon(points = polygon1, p = p)):
               potentials.update({j:"SI"})
           else:
               potentials.update({j:"NO"})
               
           
           
           
           
           
           '''
           power =  exps[exp][i]['powerH'].shape[0]
               
           potentials.update({j:{"err_s":np.abs(esfera_c[0] + 7.5 -esf_s_arr[idx_sphere_f]),
                                 "err_h":np.abs(esfera_c[3] - (esf_h_arr[idx_sphere_f] -2.9)),
                                 "med_h":esfera_c[3]-esfera_c[1]}})
           
       exps[exp][i].update({"meds":potentials})
       
           '''
       
       exps[2][list(exps[2].keys())[i]].update({"meds":potentials})
       ''' 
       if( len(exps[exp][i]['esfera']) != 0): 
           max_power_sph_idx = rpower_sphere.index(max(rpower_sphere)) 
           max_sph_refer = list(exps[exp][i]['esfera'].keys())[max_power_sph_idx]
           max_range =  exps[exp][i]['esfera'][max_sph_refer][2]
           max_perf = exps[exp][i]['esfera'][max_sph_refer][1]
          
           
           esfera_max = get_coords(max_perf, max_range)
           ax.add_patch(Polygon(esfera_max[2],edgecolor = 'black', facecolor = 'green', fill=True))
          
           plt.plot(esfera_max[0], esfera_max[1],'xw', markersize = 12)
       '''
           
       si_cta = 0
       dr_cta = 0
       
       #count_xp += 1
       '''
       
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
       '''
       
       scale_img = mpimg.imread('scale.png')
       imagebox = OffsetImage(scale_img, zoom=0.5)
       ab = AnnotationBbox(imagebox, (60, 30))
       ax.add_artist(ab)
       plt.xlabel("S [m]")
       plt.ylabel("H [m]")
       plt.grid()
       plt.draw()
       plt.plot()
       plt.xticks([0,40,80,120,160,200,240,280])
       plt.yticks([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140])
       plt.title("Echoes for " + exps[2][list(exps[2].keys())[i]]["time"])
       title = exps[2][list(exps[2].keys())[i]]["time"].replace(':','')
       plt.savefig(r'C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\Plots_exp_2'+'//'+list(exps[2].keys())[i][-11:-7]+'//'+title+'.png')
           
       #count_xp += 1
            
    except:
        pass


