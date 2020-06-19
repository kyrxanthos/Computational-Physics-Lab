#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:41:07 2019

@author: lysi2
"""
import numpy as np          #import numpy package
import sys
import math
#import matplotlib.pyplot as plt    # Import the plot package as plt 
from pandas import DataFrame


def x_function(fx,fy,ax,ay,ph,t,n):         #definition of x function
        
    x= ax*np.cos(2*math.pi*fx*t)    #use the numpy function for cos so that all the values are stored in arrays
        
    return x              #function returns the value x so that it can be accessed later

def y_function(fx,fy,ax,ay,ph,t,n):         #definition of y functiom
    
    y= ay*np.sin(2*math.pi*fy*t+ph)
    
    return y

def z_function(x,y):        #definition of z function
    
    z= x+y
    
    return z


def main():         #the main execution will be performed here, using the above functions
    
    ax = float(sys.argv[1])         #input values
    ay = float(sys.argv[2])  
    fx = float(sys.argv[3])  
    fy = float(sys.argv[4]) 
    ph = float(sys.argv[5])  
    dt = float(sys.argv[6])  
    n = float(sys.argv[7])  
  
    t = np.arange(0.0, n*dt, dt)    #create a numpy array that stores all the t values with the correct incriments
    
    x= x_function(fx,fy,ax,ay,ph,t,n)   #the functions are called
    
    y= y_function(fx,fy,ax,ay,ph,t,n)
    
    z= z_function(x,y)
    
   
    df_x = DataFrame(x)     #create a dataframe to store the values
    
    df_y = DataFrame(y)
    
    df_z = DataFrame(z)
    
    df_t = DataFrame(t)
    
    #these functions export the values from the dataframes to CSV files
    
    export_csv_x= df_x.to_csv(r'/Users/lysi2/Documents/UNI_Caltech/Ph_20_kyriacos/A2/Py_Programs/x_export.csv', index = None, header=False)
    
    
    export_csv_y= df_y.to_csv(r'/Users/lysi2/Documents/UNI_Caltech/Ph_20_kyriacos/A2/Py_Programs/y_export.csv', index = None, header=False)


    export_csv_z= df_z.to_csv(r'/Users/lysi2/Documents/UNI_Caltech/Ph_20_kyriacos/A2/Py_Programs/z_export.csv', index = None, header=False)
    
    export_csv_t= df_t.to_csv(r'/Users/lysi2/Documents/UNI_Caltech/Ph_20_kyriacos/A2/Py_Programs/t_export.csv', index = None, header=False)


    #call the above functions
    
    export_csv_x
    
    export_csv_y
    
    export_csv_z
    
    export_csv_t

main()      #close the main function
