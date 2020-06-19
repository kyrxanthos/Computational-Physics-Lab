#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:00:09 2019

Ph20 Assignment 1

@author: Kyriacos Xanthos
"""

""" Exercise 1 """

import math

import matplotlib.pyplot as plt    # Import the plot package as plt 


from pandas import DataFrame



def x_function(fx,fy,ax,ay,ph,t,n):      #definition of x function
    
    x= ax*math.cos(2*math.pi*fx*t)      #use of the math package
    
    return x

def y_function(fx,fy,ax,ay,ph,t,n):      #definition of x function
    
    y= ay*math.sin(2*math.pi*fy*t+ph)
    
    return y

def z_function(x,y):                     #definition of x function
    
    z= x+y
    
    return z


def main():       #the main execution will be performed here, using the above functions
    
    x_data = []              # lists to hold the data
    y_data = []
    z_data = [] 
    t_data = []
    
    
    #fx= float(input('Input fx: '))
    fx= 1                               #input values
    #fy= float(input('Input fy: '))
    fy = 1
#    ax= float(input('Input ax: '))
    ax=1
#    ay= float(input('Input ay: '))
    ay=1
 #   ph= float(input('Input ph: '))
    ph=0
 #   dt= float(input('Input dt: '))
    dt =0.01

  #  n= int(input('Input N: '))
    n=1000
    
    t= -dt      #negative so that the list begins from zero
    
    x= x_function(fx,fy,ax,ay,ph,t,n)   #call the functions
    
    y= y_function(fx,fy,ax,ay,ph,t,n)
    
    z= z_function(x,y)
    
    
    
    

    for i in range(0,n):       # polulate the lists with x and t
        t+=dt           #incriment t by dt
        x=x_function(fx,fy,ax,ay,ph,t,n)
        y=y_function(fx,fy,ax,ay,ph,t,n)
        z=z_function(x,y)
        t_data.append(t)    #append all the lists
        x_data.append(x)
        y_data.append(y)
        z_data.append(z)
        



    plt.plot(y_data,x_data, color = 'r')   # Default plot
    plt.title("Lissajous figures X(t) against Y(t)")  # Add title/lables
    plt.xlabel("Y(t)")
    plt.ylabel("X(t)")
    plt.savefig('trial.png')
    plt.show()                    # show the actual plot

      

    plt.plot(t_data,z_data, color = 'r')   # Default plot
    plt.title("Z(t) against t")  # Add title/lables
    plt.xlabel("t")
    plt.ylabel("Z(t)")
    plt.savefig('trial5.png')
    plt.show()                    # show the actual plot
 
    
    
    df_x = DataFrame(x_data)     #create a dataframe to store the values
    
    df_y = DataFrame(y_data)
    
    df_z = DataFrame(z_data)
    
    #these functions export the values from the dataframes to CSV files

    
    export_csv_x= df_x.to_csv(r'/Users/lysi2/Documents/UNI_Caltech/Ph_20/A1/CSV Exports/x_export.csv', index = None, header=True)
    
    
    export_csv_y= df_y.to_csv(r'/Users/lysi2/Documents/UNI_Caltech/Ph_20/A1/CSV Exports/y_export.csv', index = None, header=True)


    export_csv_z= df_z.to_csv(r'/Users/lysi2/Documents/UNI_Caltech/Ph_20/A1/CSV Exports/z_export.csv', index = None, header=True)

    
    #call the above functions

    
    export_csv_x
    
    export_csv_y
    
    export_csv_z

    print("this is a new change")
    
    


main()        #close the main function
