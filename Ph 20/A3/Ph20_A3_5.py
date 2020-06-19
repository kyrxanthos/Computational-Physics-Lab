#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:40:58 2019

@author: lysi2
"""

import numpy as np


import matplotlib.pyplot as plt    # Import the plot package as plt 


N =100

func = lambda x : np.exp(x) # function

exact = np.e -1

N_data = np.arange(0, N-1, 1)     #create a numpy array with all the N's

nlist = np.arange(0, N-1, 1)     #create a numpy array with all the N's

error = np.zeros(len(nlist))



def int_simps(func, n):   #simps function

    x = np.linspace(0, 1, 2*n + 1)  #store x values in an array
    y = func(x)
    dx = 1./n
    y[2:2*N:2] *= 2
    y[1::2] *= 4

    #simpson's formula
    area_s = dx/3.0 * np.sum(y)
    return area_s



for i, n in enumerate(nlist):
    error[i] = abs (int_simps(func, n) - exact)
    


plt.plot(N_data, error , color = 'r')   # Default plot
plt.title("Convergence rate of error of Simpson's rule")  # Add title/lables
plt.xlabel("N")
plt.ylabel("Error")
plt.show()                    # show the actual plot

    
plt.plot(N_data, error , color = 'r')   # Default plot
plt.yscale("log")
plt.xscale("log")
plt.title("Logarithmic convergence rate of error of Simpson's rule")  # Add title/lables
plt.xlabel("N")
plt.ylabel("log (Error)")
plt.show()                    # show the actual plot

    

    

