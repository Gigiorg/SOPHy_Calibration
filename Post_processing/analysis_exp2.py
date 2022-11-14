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
        fig, ax = plt.subplots(figsize = (11,9))
                        
        perf = df["Azimuth"] == i
        perf = df[perf]
        exp_tick = df['Exp Constant [dB]'][0]          
                      
        c_initial = perf['C_initial [dB]']
        c_after = perf["C_after Wr [dB]"]
        c_after_wb = perf["C_after Wb [dB]"]
        
        c_after_wb_lin = perf["C_after Wb"]
            

              
        std_initial = np.std(c_initial, ddof=1)
        std_after_wr = np.std(c_after, ddof=1)
        std_after_wb = round(np.std(c_after_wb, ddof= 1),2)
                      
                      
        date_time = pd.to_datetime(perf["Datetime"]) 
        
        theo_constant = perf['Exp Constant [dB]']
        theo_constant_lin = perf["Exp Constant"]
                      
        #Mean 
        avg_c_after_wb = np.mean(c_after_wb_lin)
        avg_c_after_wb_db = 10*np.log10(avg_c_after_wb)*np.ones(len(theo_constant))
        
        
        #print(avg_c_after_wb)
                      
        ax.scatter(date_time,c_after_wb, marker="x", linewidths=4)
        ax.plot(date_time,theo_constant, linestyle='dashed', markersize = 12, linewidth = 3, color = "red")
        ax.plot(date_time,avg_c_after_wb_db,linestyle='dashed', markersize = 12, linewidth = 3)
        #ax.scatter(date_time,c_after_wb_lin)
        #ax.plot(date_time,theo_constant_lin)
        
        
        
        ax.grid()
                      
        ax.set_title(f"Experiment 2, Azimuth: {i}°")
        ax.legend([ "Theoretical Constant",f"Fitted Experimental Constant, Offset: {round(exp_tick-10*np.log10(avg_c_after_wb),2)} dB" ,
                   f"Experimental Constant with Wr and Wb, Std: {std_after_wb}"])
      
        ax.set_xlabel("Time")
        ax.set_ylabel("Experimental RCC [dB]")
        
        ax.set_yticks([10*np.log10(avg_c_after_wb), exp_tick,60, 65, 70, 75, 80, 85])
      
                      
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