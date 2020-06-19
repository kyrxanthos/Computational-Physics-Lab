#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:24:29 2019

@author: lysi2
"""

import matplotlib.pyplot as plt    # Import the plot package as plt 

import numpy as np

import sys

def main():
    
    #import CSV files
    
    x_data = np.genfromtxt('/Users/lysi2/Documents/UNI_Caltech/Ph_20_kyriacos/A2/Py_Programs/x_export.csv', delimiter=' ', names=True)
    

    y_data = np.genfromtxt('/Users/lysi2/Documents/UNI_Caltech/Ph_20_kyriacos/A2/Py_Programs/y_export.csv', delimiter=' ', names=True)
    

    z_data = np.genfromtxt('/Users/lysi2/Documents/UNI_Caltech/Ph_20_kyriacos/A2/Py_Programs/z_export.csv', delimiter=' ', names=True)
    
    t_data = np.genfromtxt('/Users/lysi2/Documents/UNI_Caltech/Ph_20_kyriacos/A2/Py_Programs/t_export.csv', delimiter=' ', names=True)



    #if length 1 then plots x-y, if length 2 plots z-t
    
    if  len(sys.argv[1]) == 1:
        
            plt.plot(y_data,x_data, color = 'r')   # Default plot
            plt.title("Lissajous figures X against Y")  # Add title/lables
            plt.xlabel("Y")
            plt.ylabel("X")
            plt.savefig('x_y.png')
            plt.show()                    # show the actual plot
    
    elif len(sys.argv[1]) == 2:
    
    
            
        plt.plot(t_data, z_data , color = 'r')   # Default plot
        plt.title("Lissajous figures Z against t")  # Add title/lables
        plt.xlabel("t")
        plt.ylabel("z")
        plt.savefig('z_t.png')
        plt.show()                    # show the actual plot

    else:
        print("Insert correct argument")

    
main()
