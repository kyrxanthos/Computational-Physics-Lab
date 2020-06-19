"""
Ph21 A5

Kyriacos Xanthos

"""

import numpy as np
import matplotlib.pyplot as plt
import dynesty
from dynesty import plotting as dyplot


def generator(n0x_k, alpha, beta, d):
    # n0 x_k is number of points
    angles = np.random.random(n0x_k) * np.pi - np.pi / 2
    x_k = beta * np.tan(angles) + alpha
    return x_k

def probability_x(d, alpha, beta):
    
    # we take a sample max and min value of x
    x_max = np.arctan((d + 0.05 - alpha) / beta)
    x_min = np.arctan((d - 0.05 - alpha) / beta)
    x_diff = np.abs(x_max - x_min)
    # this is unnormalized
    return x_diff



def log_likelihood(rounded_data, alpha, beta):

    #logarithmic likelyhood for all x_k 's
    log_like = np.sum(np.log(np.array([probability_x(d, alpha, beta) for d in rounded_data])))

    return log_like


    
def plot(n, alpha, beta, dlogz_val=.1, interloper=False, d=1):
    x_k = np.round(generator(n, alpha, beta, d), 2) #round up
    
    #append values if interloper is true
    if (interloper == True):
        interloper_x_k= np.round(generator(n, alpha + 3, beta, d), 2)
        x_k = np.append(x_k, interloper_x_k) 
        
    #feed parameters for alpha and beta
        
    def lighthouse_logl(params):

        return log_likelihood(x_k, params[0], params[1])
    
    
    #prior transform, using documentation for range -1000, 1000
    
    def prior_transform(u):
        return [2000 * u[0] - 1000, 1000 * u[1]]
    
    ndim = 2
    sampler = dynesty.NestedSampler(lighthouse_logl, prior_transform, ndim, bound='single', nlive=200) 
    sampler.run_nested(dlogz=dlogz_val, print_progress=False)
    results = sampler.results
    
    #runplot
    
    dyplot.runplot(results)
    #plt.savefig('int_run_plot' + str(dlogz_val)+".png", dpi =300)
    plt.show()
    
    
    #cornerpoints plot
    
    fig = plt.subplots(1, 1, figsize=(10, 10))
    dyplot.cornerpoints(results, fig=fig, cmap = 'plasma',truths=np.zeros(ndim), kde=False )
    fig[1].set_ylabel('$\\beta$')
    fig[1].set_xlabel('$\\alpha$')
    plt.tight_layout()
    plt.xlim(-10, 10)
    plt.ylim(0, 10)
    plt.savefig('int_corner1' + str(dlogz_val)+".png", dpi =300)
    plt.show()
    
    
    #traceplot
    
    fig = plt.subplots(2, 2, figsize=(15, 10))
    dyplot.traceplot(results, fig=fig,truth_color='black',trace_cmap='viridis', 
                     connect=True, connect_highlight=range(5), show_titles=True )
    fig[1][1, 1].set_xlabel('$\\beta$')
    fig[1][0, 1].set_xlabel('$\\alpha$')
    fig[1][1, 0].set_ylabel('$\\beta$')
    fig[1][0, 0].set_ylabel('$\\alpha$')
    plt.tight_layout()
    #plt.savefig('int_trace' + str(dlogz_val)+".png", dpi =300)
    plt.show()

    print(results.samples[-1])
    

plot(100, 0, 3)

#plot(100, 0, 3, interloper = True)






