#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:52:55 2020

@author: lysi2
"""


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,15,500)

y_1 = np.sin(x)

y_2 = np.sin(x-10)

plt.figure(figsize = (9.5, 4.1))
plt.plot(x, y_1, label='sin(x)', color ='r', linewidth = 1)
plt.legend()
plt.plot(x, y_2, label='sin(x-1)', color ='b', linewidth = 1)
plt.legend()
plt.grid('True')
plt.xlabel('k')
plt.ylabel('Fourier coefficient')
plt.title('Fourier coefficients for a delta function convolution')
#plt.savefig("convolution.png", dpi = 300 )
plt.show()