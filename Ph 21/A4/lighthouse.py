
"""
Ph21 A4b

Kyriacos Xanthos


"""


import numpy as np
import matplotlib.pyplot as plt


def cauchy_dist(x, alpha, beta):
    
    den = np.power(beta, 2.) + np.power((x - alpha), 2.)
    return beta / den

def light_1d_post(N, true_a, true_b):

    alpha = np.linspace(-10., 10., 100)
    log_prior = np.zeros(len(alpha))
    x_k = np.array([])

    for i in range(N):
        x_k = np.array([])
        theta = np.random.random() * np.pi
        
        if theta > np.pi / 2:
            theta -= np.pi 

        x = true_a + true_b * np.tan(theta)
        x_k = np.append(x_k, x)
        log_post = log_prior + np.log(cauchy_dist(x, alpha, true_b))
        log_prior = log_post
 
    if N != 0:
        
        log_prior -= log_prior.max()   
    
    prior = np.exp(log_prior)
    mean_val = np.mean(x_k.mean())
    print(mean_val)
    return alpha, prior, x_k.mean()

def light_2d_post(N, true_a, true_b):

    a = np.linspace(-10., 10., 100)
    b = np.linspace(0., 10., 100)
    log_prior = np.zeros((len(a), len(a)))
    alpha, beta = np.meshgrid(a, b)


    for n in range(N):
        theta = np.random.random() *np.pi

        if theta > np.pi / 2:
            theta -= np.pi 

        x = true_a + true_b * np.tan(theta)
        log_post = log_prior + np.log(cauchy_dist(x, alpha, beta))
        log_prior = log_post

    if N != 0:
        log_prior -= log_prior.max()

    prior = np.exp(log_prior)
    return alpha, beta, prior

def plot_1d(N, true_a, true_b):

    plt.figure(figsize = (15, 10))
    plt.suptitle('posterior distributions for various values n' , fontsize = 25,\
    ha = 'center', fontweight='bold')

    for i, n in enumerate(N):
        alpha, density, mean= light_1d_post(n, true_a, true_b)
        plt.subplot(4, 4, i+1)
        plt.axvline(mean, color='r')
        plt.plot(alpha, density, color = 'b')
        plt.title('n = ' + str(n))
        plt.xlabel('$\\alpha$')
        plt.ylabel('Posterior Density')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.savefig("1d_plot.png", dpi =300)
    plt.show()
    
    



def plot_2d(N, true_a, true_b):

    plt.figure(figsize = (15, 10))
    plt.suptitle('posterior distributions for various values n' , fontsize = 25,\
    ha = 'center', fontweight='bold')

    for i, n in enumerate(N):
        alpha, beta, density = light_2d_post(n, true_a, true_b)
        plt.subplot(4, 4, i+1)
        #plt.imshow(density, cmap='hot')
        plt.contour(alpha, beta, density)
        plt.title('n = {}'.format(n))
        plt.xlabel('$\\alpha$')
        plt.ylabel('Posterior Density')
        

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.savefig("2d_plot.png", dpi =300)
    plt.show()
    


N = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]  #different values of N

true_a = 4
true_b = 2

plot_1d(N, true_a, true_b)
plot_2d(N, true_a, true_b)


