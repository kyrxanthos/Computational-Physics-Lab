"""
Ph20 Assignment 3 
Created By Kyriacos Xanthos

"""

#QUESTIONS 2,3, 7

import numpy as np 
import math


from scipy.integrate import simps
from scipy.integrate import quad
from scipy.integrate import romberg

from numpy import trapz




def e_function(z):
    return math.e**z



def main():
    
    a = 0
    b = 1
    N = 100
    
    exact = np.exp(1) -1    #exact value for the integral

    
    z = np.arange(a, b, (1/N))       #store x values in an array
    
    y_2 = e_function(z)
    
    

    # using Trapezoidal rule:

    area = trapz(y_2,z)
    print ('area using Trapezium rule: ',area)
    print ('approximate error: ', exact - area)


    # using Simpson's rule:

    area = simps(y_2,z)
    print ('area using Simpsons rule: ',area)
    print ('approximate error: ', exact - area)
    
    # using quad:
    
    func = lambda x: np.exp(x)
    area = quad(func, a, b)
    print ('area using quad: ',area)
    print ('approximate error: ', exact - area)

    
    # using romberg:

    area = romberg(func, a, b)
    print ('area using romberg: ',area)
    print ('approximate error: ', exact - area)

    





    

main()
