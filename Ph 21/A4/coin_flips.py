
"""
Ph21 A4a

Kyriacos Xanthos

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

def uniform(no_H):
    "no_H = number of H values"
    return 1./no_H * np.ones(no_H)

def gaussian(H, mu, sigma):
    "calculates the gaussian given the above parameters"
    gauss = 1./(np.sqrt(2.*np.pi)*sigma) * \
    np.exp(-np.power((H-mu), 2.) / (2. * np.power(sigma, 2.)))
    return gauss

def run_tests(N, real_h, prior_dist, mu=0.5, sigma=0.4):

    H = np.linspace(0.0, 1., 400)
    h = 0.  #heads 
    
    if prior_dist.lower() == 'uniform':
        prior = uniform(len(H))
    
    else: 
        prior = gaussian(H, mu, sigma)

    for n in range(1, N+1):
        flip = np.random.random()
        
        if flip < real_h:
            h += 1.
        
        likelihood = sp.comb(n, h) * np.power(H, h) * np.power((1-H), (n-h))
        posterior = likelihood * prior
        prior = posterior
    
    prior = prior / sum(prior)
    return H, prior

def plot(N, real_h, prior_dist, mu=0.5, sigma=0.2):

    plt.figure(figsize = (15, 10))
    plt.suptitle(str(prior_dist) +', H = ' + str(real_h), fontsize = 25,\
    ha = 'center', fontweight='bold')

    for i, n in enumerate(N):
        H, res = run_tests(n, real_h, prior_dist, mu, sigma)
        plt.subplot(4, 4, i+1)
        plt.plot(H, res, color = "b")
        plt.title('n =  '+ str(n))
        plt.xlabel('H')
        plt.ylabel('Posterior Density')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.savefig(str(real_h) + "_" + str(prior_dist) +".png", dpi =300)
    plt.show()


N = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]  #different values of N

real_h_val = [0.5, 0.6, 0.7] # different values of h

for i in real_h_val:
    
    plot(N, i, 'uniform')
    plot(N, i, 'gaussian')

