"""
Ph21 A5

Kyriacos Xanthos

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import dynesty



def generator(H, n_values):  #generates random coin flips and takes the sum
    coin_flips = np.random.random(n_values) < H
    tot_sum = np.sum(coin_flips)
    return tot_sum


def like_func(n, h, H):  #calculates the likelihood
    if (H <= 0 or H >= 1): #ensure H is less than 1 and non negative
        return 0
    likelihood = sp.comb(n, h) * np.power(H, h) * np.power((1-H), (n-h))
    
    return likelihood

def uniform_prior(H):
    return H

def gaussian(H, mu = 0.5, sigma = 0.1):
    "calculates the gaussian given the above parameters"
    gauss = 1./(np.sqrt(2.*np.pi)*sigma) * \
    np.exp(-np.power((H-mu), 2.) / (2. * np.power(sigma, 2.)))
    return gauss


def plot(N1, prior_dist, H, dlogz_val, nlive_val,  **prior_kwargs):

    plt.figure(figsize = (15, 10))
    plt.suptitle(' dlogz = ' + str(dlogz_val) + ', nlive = ' \
                 + str(nlive_val), fontsize = 25, ha = 'center')
        
    #prior transform (constant)
        

    for i, n_values in enumerate(N1):
        
        sum_val = generator(H, n_values) #total sum of Hval
        
        def log_likelyhood(H):
            return np.log(like_func(n_values, sum_val, H[0]))
        
        #same as found in documentation:
        
        ndim = 1 #number of dimensions in the problem
        sampler = dynesty.NestedSampler(log_likelyhood, prior_dist, ndim, \
                                        bound='single', nlive=nlive_val) 
            
        sampler.run_nested(dlogz=dlogz_val, print_progress=False)
        results = sampler.results
        

        x_axis = results.samples
        y_axis = np.exp(results.logl)
        
        plt.subplot(4, 4, i+1)
        plt.plot(x_axis, y_axis, '.', color = "b")
        plt.title('n_values =  '+ str(n_values))
        plt.xlabel('likelihood')
        plt.ylabel('N0 of heads')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.savefig( "gaussian_0.5_0.2" + str(dlogz_val) +str(nlive_val) +".png", \
     #           dpi =300, bbox_inches = 'tight')
    plt.show()





N1 = [4, 8, 16, 32, 64, 128, 256, 512]



plot(N1, uniform_prior, H = 0.5, dlogz_val = 1, nlive_val = 10)
#plot(N1, uniform_prior, H = 0.5, dlogz_val = 1, nlive_val = 30)

#plot(N1, uniform_prior, H = 0.5, dlogz_val = 0.1 , nlive_val = 10)
#plot(N1, uniform_prior, H = 0.5, dlogz_val = 0.01, nlive_val = 10)


plot(N1, gaussian, H = 0.5, dlogz_val = 1, nlive_val = 10)
#plot(N1, gaussian(0.5, 0.5, 0.2), H = 0.5, dlogz_val = 1, nlive_val = 10)







