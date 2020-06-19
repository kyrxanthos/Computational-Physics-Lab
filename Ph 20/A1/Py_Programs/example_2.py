#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:30:17 2019

@author: lysi2
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True})

x = np.random.random(10)
y = np.random.random(10)

plt.figure()
plt.plot(x, y, 'o')
plt.xlabel('my xlabel')
plt.ylabel('my ylabel')
plt.show()

