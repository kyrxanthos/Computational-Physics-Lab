"""
Ph20 Assignment 3 
Created By Kyriacos Xanthos

"""

#QUESTION 4 Trapezium using for loop

import numpy as np 
import matplotlib.pyplot as plt    # Import the plot package as plt 


from scipy.integrate import simps



#Integration using trapezium rule
    


def int_trapz(func, a, b, n):
    x = np.linspace(a, b, n+1)
    y = func(x)

    y_right = y[1:] 
    y_left = y[:-1] 
    dx = (b - a)/n
    area_t = (dx/2) * np.sum(y_right + y_left)
    return area_t
    

def integration_s(func ,n):
    x = np.arange(0, 1, (1/n))  #make an array from 0 to 1 with 1/n space
    y = func(x)
    area_e = simps(y,x)     #use simps for simpson's rule
    return area_e



def main():
    

    N = 50
    exact = np.e -1
    
    n = 1   #initialize n
    a =0
    b =1
    area_t = []  #create a list to hold the area data
    area_e = []  #create a list to hold the area data
    func = lambda x : np.exp(x)
    N_data = np.arange(0, N-1, 1)     #create a numpy array with all the N's
    

    
    for i in range (1 , N):     #loop for every single N

        n =+ i
        q =int_trapz(func, a, b, n)
        p = integration_s(func,n)
        ans = abs (q - exact)   #difference
        ans2 = abs (p - exact)
        area_t.append(ans)   #append for every value
        area_e.append(ans2)
    print(area_e)
        
    
    #print(area_data)
        
    plt.plot(N_data, area_t , color = 'r')   # Default plot
    plt.title("Convergence rate of error Trapezium")  # Add title/lables
    plt.xlabel("N")
    plt.ylabel("Error")
    plt.savefig('trial.png')    
    plt.show()                    # show the actual plot




main()
