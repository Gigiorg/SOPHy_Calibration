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

PATH_TABLES = r"C:\Users\GIBS\Documents\SOPHy_Calibration\Post_processing"


for root, direc, file in os.walk(PATH_TABLES):
    for file in file:
        if(file.endswith(".xlsx")):
            df = pd.ExcelFile(PATH_TABLES+'\\'+file).parse('tabla')
        
            f = len(pd.unique(df.Azimuth))
            
            if (f > 0):
                
                for i in sorted(pd.unique(df.Azimuth)):
                      fig = plt.figure(figsize = (8,6))
                        
                      perf = df["Azimuth"] == i
                      perf = df[perf]
                      
                      
                      c_initial = perf['C_initial [dB]']
                      c_after = perf["C_after [dB]"]
                      
                      std_initial = np.std(c_initial, ddof=1)
                      std_after = np.std(c_after, ddof=1)
                      
                      
                      
                      date_time = pd.to_datetime(perf["Datetime"]) 
                      
                      #print(date_time)
                      
                      ax_1 = fig.add_subplot(2,1,1)
                      ax_1.scatter(date_time,c_initial)
                      ax_1.plot(date_time, c_initial)
                      ax_1.grid()
                      ax_1.set_title(f"Std: {std_initial}")
                      ax_2 = fig.add_subplot(2,1,2)
                      ax_2.scatter(date_time,c_after)
                      ax_2.plot(date_time, c_after)
                      ax_2.grid()
                      ax_2.set_title(f"Std: {std_after}")
                      
                      
                      
                
                  
          