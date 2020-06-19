#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:41:07 2019

@author: lysi2
"""

import numpy as np          #import numpy package

import math

import matplotlib.pyplot as plt    # Import the plot package as plt 

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
    
    ax =1         #input values
    fx=1
    ay=1
    ph=math.pi/2
    fy=1
    
    dt= 0.01
    

    

    n=1000
    
    
    t = np.arange(0.0, n*dt, dt)    #create a numpy array that stores all the t values with the correct incriments
    
    x= x_function(fx,fy,ax,ay,ph,t,n)   #the functions are called
    
    y= y_function(fx,fy,ax,ay,ph,t,n)
    
    z= z_function(x,y)
    

    
    
    plt.plot(y,x, color = 'r')   # Default plot
    plt.title("Lissajous figures X against Y")  # Add title/lables
    plt.xlabel("Y")
    plt.ylabel("X")
    plt.savefig('phase.png')
    plt.show()                    # show the actual plot
    


    
    plt.plot(t,z, color = 'r')   # Default plot
    plt.title("Lissajous figures Z against t")  # Add title/lables
    plt.xlabel("t")
    plt.ylabel("z")
    plt.savefig('trial4.png')
    plt.show()                    # show the actual plot
    
    
   
    df_x = DataFrame(x)     #create a dataframe to store the values
    
    df_y = DataFrame(y)
    
    df_z = DataFrame(z)
    
    #these functions export the values from the dataframes to CSV files
    
    export_csv_x= df_x.to_csv(r'/Users/lysi2/Documents/UNI_Caltech/Ph_20/A1/CSV Exports/x_export.csv', index = None, header=True)
    
    
    export_csv_y= df_y.to_csv(r'/Users/lysi2/Documents/UNI_Caltech/Ph_20/A1/CSV Exports/y_export.csv', index = None, header=True)


    export_csv_z= df_z.to_csv(r'/Users/lysi2/Documents/UNI_Caltech/Ph_20/A1/CSV Exports/z_export.csv', index = None, header=True)

    #call the above functions

    
    export_csv_x
    
    export_csv_y
    
    export_csv_z
    
    

    
main()      #close the main function
