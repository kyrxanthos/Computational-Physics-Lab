"""
Ph20 Assignment 4
Created By Kyriacos Xanthos

"""

import numpy as np
from matplotlib import pyplot as plt
import scipy
from numpy import array
from scipy import optimize
from scipy import integrate
from scipy import stats



# Part 4
def linear_func(redshift, luminocity_dist, luminocity_unc, light_speed):
    small_redshift = []
    small_lum = []
    small_lum_unc = []
    
    for i in range(len (redshift)):
        if redshift[i] < 0.05:
            small_redshift.append(redshift[i])
            small_lum.append(luminocity_dist[i])
            small_lum_unc.append(luminocity_unc[i])
    

            
    array_1 = array(small_redshift)
    array_2 = array(small_lum)
    array_3 = array(small_lum_unc)
    
    coef = np.polyfit(array_1,array_2,1)
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    poly1d_fn = np.poly1d(coef) 



    
    plt.figure(figsize = (9.5, 4.1))
    plt.plot(array_1,array_2, 'yo', array_1, poly1d_fn(array_1), '-')
    plt.errorbar(array_1,array_2,yerr =array_3 ,fmt='.',lw =0.3, elinewidth = 0.3, ecolor = 'r')
    plt.xlabel("Small Redshift")
    plt.ylabel("Luminocity distance (Mpc)")
    #plt.savefig("3rd.png", dpi = 1000 )
    plt.show()
    


    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(small_redshift, small_lum)
    
    
    H0 = light_speed / slope
    
    print ("H0 is " , H0)
    return slope
    
# Part 5
    
def func(redshift,slope,q): 
	return slope*redshift*(1+((1-q)/2)*redshift)



def non_linear_func(light_speed, redshift, luminocity_dist, luminocity_unc, array_5):
    
    paramnl,_ = optimize.curve_fit(func, redshift, luminocity_dist)
    gradient = paramnl[0]
    q = paramnl[1]

    
    H0nl = light_speed/gradient
    print("Non-Linear Hubble is: ", H0nl)

    
    
    plt.figure(figsize = (9.5, 4.1))
    plt.plot(array_5, func(array_5, gradient, q), label='Linear Fit', color ='r', linewidth = 1)
    plt.errorbar(redshift,luminocity_dist, yerr = luminocity_unc,fmt='.',lw =0.3, elinewidth = 0.3, ecolor = 'g',label='Data')
    plt.xlabel('Redshift')
    plt.ylabel('Luminosity Distance (Mpc)')
    #plt.savefig("4th.png", dpi = 1000 )
    plt.show()

# Part 6

def FLRW_Integral(x,g,om):
	integrand = 1/np.sqrt(((1+x)**3)*om+(1-om))
	return g*integrate.cumtrapz(integrand, x=x, initial = 0)*(1+x)  #try diff int and check

def FLRW(light_speed, redshift, luminocity_dist, array_5, luminocity_unc):
    
    param_FLRW , p_cov = optimize.curve_fit(FLRW_Integral, redshift, luminocity_dist)
    gradient_2 = param_FLRW[0]
    om = param_FLRW[1]
    error = np.sqrt(p_cov[1][1])
    sigma =(1-om)/error
    H0_FLRW = light_speed/gradient_2
    
    print("FLRW Hubble is: ", H0_FLRW)
    print("Omega is: ", om)
    print ("Error is ", error)
    print("Sigma is ", sigma)


    
    
    plt.figure(figsize = (9.5, 4.1))
    plt.plot(array_5, func(array_5, gradient_2, om), label='Linear Fit', color ='r', linewidth = 1)
    plt.errorbar(redshift,luminocity_dist, yerr = luminocity_unc,fmt='.',lw =0.3, elinewidth = 0.3, ecolor = 'g',label='Data')
    plt.xlabel('Redshift')
    plt.ylabel('Luminosity Distance (Mpc)')
    #plt.savefig("5th.png", dpi = 1000 )
    plt.show()
    
    
    
    
    
    

def main():
    
    light_speed = 299792.458
    
    array_5 = np.linspace(0, 1.5, 100) # sample lin  space of ordered redshift
    
    data = np.genfromtxt('SCPUnion2.1_mu_vs_z.txt', comments='#', delimiter='	')
    
    redshift = data[:,1]
    distance_modulus = data[:,2]
    distance_moduluts_unc = data[:,3]
    
    luminocity_dist = (10**((distance_modulus/5)+1))/(10**6)
    luminocity_unc = (np.log(10)/5)*(luminocity_dist)*distance_moduluts_unc
    
    #Plot of Distance against Redshift
    
    
    plt.figure(figsize = (9.5, 4.1))
    plt.errorbar(redshift,distance_modulus,yerr =distance_moduluts_unc,fmt='.',lw =0.3, elinewidth = 0.3, ecolor = 'r')
    plt.xlabel("Redshift")
    plt.ylabel("Distance Modulus")
    #plt.savefig("1st.png", dpi = 1000 )
    plt.show()
    
    #Plot of Luminocity distance against Redshift
    
    plt.figure(figsize = (9.5, 4.1))
    plt.errorbar(redshift,luminocity_dist,yerr =luminocity_unc,fmt='.',lw =0.3, elinewidth = 0.3, ecolor = 'r')
    plt.xlabel("Redshift")
    plt.ylabel("Luminocity distance (Mpc)") 
    #plt.savefig("2nd.png", dpi = 1000 )
    plt.show()
    
    linear_func(redshift, luminocity_dist, luminocity_unc, light_speed)
    non_linear_func(light_speed, redshift, luminocity_dist, luminocity_unc, array_5)
    FLRW(light_speed, redshift, luminocity_dist, array_5, luminocity_unc)
    

    
main()

