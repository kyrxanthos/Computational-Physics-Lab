#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:23:55 2020

@author: lysi2
"""

import numpy as np
from scipy.stats.mstats import zscore
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pca_code(data):
    #raw_implementation
    var_per=.98
    data-=np.mean(data, axis=0)
    
    cov_mat=np.cov(data, rowvar=False)
    evals, evecs = np.linalg.eigh(cov_mat)
    idx = np.argsort(evals)[::-1] #sorts them in order
    evecs = evecs[:,idx]
    evals = evals[idx]
    
    variance_retained=np.cumsum(evals)/np.sum(evals)
    index=np.argmax(variance_retained>=var_per)
    evecs = evecs[:,:index+1]
    print("_"*40)
    print("eigenvalues:", evals)
    print("eigenvectors:" , evecs.T[0, :])
    print("_"*40)

    
"""2d data"""
dependent1 = np.linspace(0., 20., 200)
error_x = np.random.uniform(low=0, high=30, size=np.size(dependent1))
error_y = np.random.uniform(low=0, high=30, size=np.size(dependent1))
ind = 5 * dependent1 + error_x 
data1 = np.array([ind, dependent1]).T

pca_code(data1) #run

"""3d data """


dependent2 = np.linspace(0., 20., 500)
error_x1 = np.random.uniform(low=0, high=50, size=np.size(dependent2))
error_x2 = np.random.uniform(low=0, high=50, size=np.size(dependent2))
error_x3 = np.random.uniform(low=0, high=50, size=np.size(dependent2))


error_y1 = np.random.uniform(low=0, high=50, size=np.size(dependent2))
error_y2 = np.random.uniform(low=0, high=50, size=np.size(dependent2))
error_y3 = np.random.uniform(low=0, high=50, size=np.size(dependent2))



dependent2_1 = dependent2 + error_y1
dependent2_2 = dependent2 + error_y2
dependent2_3 = dependent2 + error_y3


ind1 = 5 * dependent2_1 +error_x1
ind2 = -2 * dependent2_2 +error_x2
ind3 = 3 * dependent2_3 +error_x3 + dependent2_2


data2 = np.array([ind1, dependent2_1, ind2, dependent2_2,ind3, dependent2_3]).T

pca_code(data2) #run


""" Plots """



plt.figure(figsize = (15, 10))
plt.scatter(ind, dependent1, color='b', marker = '.')
plt.title('2d data')
plt.xlabel('x')
plt.ylabel('y')
#plt.savefig("2d_plot.png", dpi = 300)
plt.show()

# Plot spring data
plt.figure(figsize = (15, 10))
plt.scatter(ind1, dependent2_1, color='b', marker = '.')
plt.scatter(ind2, dependent2_2, color='black', marker = '.')
plt.scatter(ind3, dependent2_3, color='g', marker = '.')
plt.title('3d data')
plt.xlabel('x')
plt.ylabel('y')
#plt.savefig("2-3d_plot.png", dpi = 300)
plt.show()

def plot_figs(fig_num, elev, azim):
    fig = plt.figure(fig_num, figsize=(7, 5))
    plt.title('3d data')
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=elev, azim=azim)

    ax.scatter(ind1[::10], ind2[::10], ind3[::10], marker='.',color = 'r', alpha=1)
    Y = np.c_[ind1, ind2, ind3]

    # Using SciPy's SVD, this would be:
    # _, pca_score, V = scipy.linalg.svd(Y, full_matrices=False)

    pca = PCA(n_components=3)
    pca.fit(Y)
    V = pca.components_

    x_pca_axis, y_pca_axis, z_pca_axis = 3 * V.T
    x_pca_plane = np.r_[x_pca_axis[:2], - x_pca_axis[1::-1]]
    y_pca_plane = np.r_[y_pca_axis[:2], - y_pca_axis[1::-1]]
    z_pca_plane = np.r_[z_pca_axis[:2], - z_pca_axis[1::-1]]
    x_pca_plane.shape = (2, 2)
    y_pca_plane.shape = (2, 2)
    z_pca_plane.shape = (2, 2)
    ax.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_title("3d data ")
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    #plt.savefig("3d_plot.png", dpi = 300)


plot_figs(1, 30, 20) #plot 3d graph
