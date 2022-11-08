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

PATH_5 = r"C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\EXP2"


df = pd.ExcelFile(PATH_5+'\\'+'Table_exp2.xlsx').parse('hits')
        
f = len(pd.unique(df.Azimuth))
           

if (f > 0):
                
    for i in sorted(pd.unique(df.Azimuth)):
        fig, ax = plt.subplots(figsize = (14,12))
                        
        perf = df["Azimuth"] == i
        perf = df[perf]
                      
                      
        c_initial = perf['C_initial [dB]']
        c_after = perf["C_after Wr [dB]"]
        c_after_wb = perf["C_after Wb [dB]"]
                      
        std_initial = np.std(c_initial, ddof=1)
        std_after_wr = np.std(c_after, ddof=1)
        std_after_wb = np.std(c_after_wb, ddof= 1)
                      
                      
        date_time = pd.to_datetime(perf["Datetime"]) 
        theo_constant = perf['Exp Constant [dB]']
                      
                      
        ax.scatter(date_time,c_initial)
        ax.plot(date_time,c_initial)
                      
        ax.scatter(date_time,c_after)
        ax.plot(date_time,c_after)
                      
    
        ax.scatter(date_time,c_after_wb)
        ax.plot(date_time,c_after_wb)
                      
        ax.plot(date_time,theo_constant, linestyle='dashed')
        ax.grid()
                      
        ax.set_title(f"Experiment 2, Azimuth: {i}°")
        ax.legend([f"Without Wr and Wb - Std: {std_initial}",f"With Wr - Std: {std_after_wr}",
                   f"With Wr and Wb Std: {std_after_wb}", "Theoretical Constant"])
      
        ax.set_xlabel("Time")
        ax.set_ylabel("Experimental RCC [dB]")
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

PATH_5 = r"C:\Users\GIBS\Documents\Documents\SOPHy_Calibration\Processing\EXP2"


#Search all the .xlsx files and iterate over each one to plot the comparative RCC



            
df = pd.ExcelFile(PATH_5+'\\'+'Table_exp2.xlsx').parse('hits')
f = len(pd.unique(df.Azimuth))

if (f > 0):
                
    for i in sorted(pd.unique(df.Azimuth)):
        
        holder = np.array([])
        c_in_2 = np.array([])
        
        perf = df["Azimuth"] == i
        perf = df[perf]
        
        wr = np.array(perf['RWF'])
        wb = np.array(perf['BWF'])
        w = wr*wb
                    
        c_in = perf['C_initial']
        c_in_db = np.array(perf['C_initial [dB]'])
                    
                    
        holder = np.append(holder,w) 
        c_in_2 = np.append(c_in_2,c_in_db)
                    
        
        plt.figure(figsize=(10,10))
        plt.scatter(holder, c_in_2)
        plt.title(f"Experiment 2 - Scatter plot WrWb vs Pr*r^4 for Azimuth: {i}°")
        plt.xlabel("WrWb")
        plt.ylabel("Pr*r^4 [dB]")
        plt.grid()
        plt.show()