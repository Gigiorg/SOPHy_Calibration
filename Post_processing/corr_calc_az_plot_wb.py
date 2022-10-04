# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:49:02 2022

@author: GIBS
"""
#%%
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Post - processing

PATH_TABLES = r"C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Post_processing\Tables_after_wr_wb"



#Search all the .xlsx files and iterate over each one to plot the comparative RCC
for root, direc, file in os.walk(PATH_TABLES):
    for file in file:
        if(file.endswith(".xlsx")):
            df = pd.ExcelFile(PATH_TABLES+'\\'+file).parse('tabla')
        
            f = len(pd.unique(df.Azimuth))
            
            if (f > 0):
                
                for i in sorted(pd.unique(df.Azimuth)):
                      fig, ax = plt.subplots(figsize = (14,12))
                        
                      perf = df["Azimuth"] == i
                      perf = df[perf]
                      
                      
                      c_initial = perf['C_initial [dB]']
                      c_after = perf["C_after [dB]"]
                      c_after_wb = perf["C_after_wb [dB]"]
                      
                      #c_after_wb = perf['Constant after Wb [dB]']
                      std_initial = np.std(c_initial, ddof=1)
                      std_after_wr = np.std(c_after, ddof=1)
                      std_after_wb = np.std(c_after_wb, ddof= 1)
                      
                      
                      date_time = pd.to_datetime(perf["Datetime"]) 
                      
                      #print(date_time)
                      
                      ax.scatter(date_time,c_initial)
                      ax.plot(date_time,c_initial)
                      
                      ax.scatter(date_time,c_after)
                      ax.plot(date_time,c_after)
                      
                      ax.scatter(date_time,c_after_wb)
                      ax.plot(date_time,c_after_wb)
                      ax.grid()
                      
                      ax.set_title(f"{file}, {i}")
                      ax.legend([f"Without Wr, Wb, std: {std_initial}",f"With Wr std: {std_after_wr}",f"With Wr and Wb std: {std_after_wb}"])
                      '''
                      ax_1 = fig.add_subplot(3,1,1)
                      ax_1.scatter(date_time,c_initial)
                      ax_1.plot(date_time, c_initial)
                      ax_1.grid()
                      ax_1.set_title(f"{file}, {i} Standard Deviation: {std_initial}")
                      ax_2 = fig.add_subplot(3,1,2)
                      ax_2.scatter(date_time,c_after)
                      ax_2.plot(date_time, c_after)
                      ax_2.grid()
                      ax_2.set_title(f"{file}, {i} Standard Deviation: {std_after_wr}")
                      ax_3 = fig.add_subplot(3,1,3)
                      ax_3.scatter(date_time,c_after_wb)
                      ax_3.plot(date_time, c_after_wb)
                      ax_3.grid()
                      ax_3.set_title(f"{file}, {i} Standard Deviation: {std_after_wb}")
                      '''
#%%

for root, direc, file in os.walk(PATH_TABLES):
    for file in file:
        if(file.endswith(".xlsx")):
            df = pd.ExcelFile(PATH_TABLES+'\\'+file).parse('tabla')
        
            f = len(pd.unique(df.Azimuth))
            
            if (f > 0):
                
                for i in sorted(pd.unique(df.Azimuth)):
                      fig = plt.figure(figsize = (10,8))
                        
                      perf = df["Azimuth"] == i
                      perf = df[perf]
                      
                      
                      c_initial = perf['C_initial [dB]']
                      c_after = perf["C_after [dB]"]
                      
                      std_initial = np.std(c_initial, ddof=1)
                      std_after = np.std(c_after, ddof=1)
                      
                      
                      
                      date_time = pd.to_datetime(perf["Datetime"]) 
                      
                      #print(date_time)
                      
                      ax_1 = fig.add_subplot(1,1,1)
                      ax_1.scatter(date_time,c_initial)
                      ax_1.plot(date_time, c_initial)
                      ax_1.scatter(date_time,c_after)
                      ax_1.plot(date_time, c_after)
                      ax_1.grid()
                      ax_1.legend([f"Std: {std_initial}",f"Std: {std_after}"])
                      ax_1.set_title(f"{file} - {i}")
                      
                      
                      
                      
                
                  
          