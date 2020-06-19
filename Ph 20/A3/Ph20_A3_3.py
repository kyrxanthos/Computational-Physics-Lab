"""
Ph20 Assignment 3 
Created By Kyriacos Xanthos

"""



import numpy as np 
import math

from scipy.integrate import simps

# QUESTION 6



def e_function(x):
    return math.e**x


def integration_s(x ,n):
    x = np.arange(0, 1, (1/n))  #make an array from 0 to 1 with 1/n space
    y = e_function(x)
    area_e = simps(y,x)     #use simps for simpson's rule
    return area_e




def extended_simpson_acc(integration_s, aq):
    err = aq + 1. #ensure first loop occurs
    N_1 = 1         #No of steps
    k  = 1         #exp factor
    prev = 0   #store the previous loop value
    
    #repeat calculations until desired accuracy achieved
    while(err > aq):
        N = (2**k)*N_1
        x = np.arange(0, 1, (1/N))

        int_curr = integration_s(x, (2**k)*N_1)
        err = abs((int_curr-prev)/int_curr)
        
        #update stored value for comparison
        prev = int_curr
        k += 1
        
    return (int_curr, N)


def main():
    
    aq = 0.02
    
    int_curr = extended_simpson_acc(integration_s, aq) #call the function
    
    print (int_curr)
    
main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    