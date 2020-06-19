
"""
Ph21 A3

Kyriacos Xanthos

This program uses different methods for 
edge detection  on different images


"""

from PIL import Image
from PIL import ImageFilter
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import filters


im = Image.open("/Users/lysi2/Documents/UNI_Caltech/Ph_21/A3/volcano.jpg")

img = cv2.imread('volcano.jpg', 0)

print(im.format, im.size, im.mode)


def general_edges(input_image):
    imageWithEdges = input_image.filter(ImageFilter.FIND_EDGES)
    return imageWithEdges



def save_image(output_image, name):
    output_image.save(fp = name, format = "JPEG")
    


#original image
    
im_edges = general_edges(im)

#save_image(im_edges, name = "original_with_edges.png")

#using different blurs
im2 = im.filter(ImageFilter.GaussianBlur(radius = 0.5)) 
im3 = im.filter(ImageFilter.GaussianBlur(radius = 0.7)) 
im4 = im.filter(ImageFilter.GaussianBlur(radius = 1)) 
im2_edges = general_edges(im2)
im3_edges = general_edges(im3)
im4_edges = general_edges(im4)

all_methods_blurs = [im_edges, im2_edges, im3_edges, im4_edges]

def all_plots_blurs(all_methods_blurs):
    for blr in all_methods_blurs:
        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(blr,cmap = 'gray')
        plt.title('Diff blur Edge Image'), plt.xticks([]), plt.yticks([])
        #plt.savefig("trial_both.png", dpi = 300)
        plt.show()
        
all_plots_blurs(all_methods_blurs)

# Part 5



sigmas = [0.1, 0.5, 1., 2.]

def gauss_der(im, sigma):
    Gx = np.zeros(im.shape);
    Gy = np.zeros(im.shape);
    # computing the x derivative
    filters.gaussian_filter(im, sigma=sigma, order=[1, 0], output=Gx,
                            mode='nearest')
    #computing the y derivative
    filters.gaussian_filter(im, sigma=sigma, order=[0, 1], output=Gy,
                            mode='nearest')
    magnitude = np.sqrt(Gx**2 + Gy**2)
    return magnitude





blurred = im.filter(ImageFilter.GaussianBlur(radius=1))
gray_blurred = np.array(blurred.convert('L'))


def diff_sigma_plots(sigmas):
    for sigma in sigmas:
        magnitude = gauss_der(gray_blurred, sigma)
        plt.imshow(magnitude,cmap = 'gray')   
        plt.title('First derivative of gaussian with width:  ' + str(sigma)), plt.xticks([]), plt.yticks([])
        #plt.savefig("gauss1_der_blur" + str(sigma) +".png", dpi = 200)
        plt.show()

diff_sigma_plots(sigmas)



# Part 6




# Canny
edges_canny = cv2.Canny(img, 100, 100)

# Sobel
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
edges_sobel = np.hypot(sobel_x, sobel_y)
edges_sobel *= 255.0 / np.max(edges_sobel)



# Laplacian
edges_laplacian = cv2.Laplacian(img, cv2.CV_64F)


# Scharr
schar_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
schar_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
edges_scharr = np.hypot(schar_x, schar_y)
edges_scharr *= 255.0 / np.max(edges_scharr)


all_methods = [edges_sobel, edges_laplacian, edges_canny, edges_scharr]



def all_plots(all_methods):
    for method in all_methods:
        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(method,cmap = 'gray')
        plt.title('Edge using Sobel'), plt.xticks([]), plt.yticks([])
        #plt.savefig("sobel.png", dpi = 300)
        plt.show()
    
all_plots(all_methods)
    