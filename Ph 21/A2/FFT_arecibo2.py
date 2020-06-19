


import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

"PART1"

T = 10

N = 10000


time = np.linspace(0, T, N)


def cos_fun(t,a, f, phi ):
    cos = a * np.cos(2*5*np.pi *f*t + phi)
    return cos


# plot of original cos
plt.figure(figsize = (9.5, 4.1))
original_cos = cos_fun(time, 2, 1/2, np.pi/4)


plt.plot(time, original_cos, color ='b')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,2)
#plt.savefig('1', dpi = 150)
plt.show()


#create general functions that calculate fft and ifft

def general_fft(y, d):
    fft = np.fft.fft(y) / len(y)
    freqs = np.fft.fftfreq(len(y), d)
    return fft, freqs

def general_ifft(fftY, l):
    ifft =  np.fft.ifft(fftY * l)
    return ifft



fft_cos, freqs_cos = general_fft(original_cos, time[1]-time[0])


# plots of fft and ifft
plt.figure(figsize = (9.5, 4.1))
plt.plot(freqs_cos, fft_cos, color ='b')
plt.xlabel('frequency')
plt.ylabel('FFT Magnitude')
plt.xlim(-10, 10)
#plt.savefig('2', dpi = 150)
plt.show()

ifft_cos = general_ifft(fft_cos, len(original_cos))

plt.figure(figsize = (9.5, 4.1))
plt.plot(time, ifft_cos, color ='b')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 2)
#plt.savefig('3', dpi = 150)
plt.show()


def gaussian_fun(t, a, b, L):
    gaussian = a * np.exp(-b * ((t - L / 2) ** 2))
    return gaussian

#plot of original gaussian

original_gaussian = gaussian_fun(time, 5, 2, 10)

plt.figure(figsize = (9.5, 4.1))
plt.plot(time, original_gaussian, color ='b')
plt.xlabel('x')
plt.ylabel('y')
#plt.savefig('4', dpi = 150)
plt.show()

fft_gaussian, freq_gaussian = general_fft(original_gaussian, time[1] - time[0])

#plot of fft, ifft gaussian
plt.figure(figsize = (9.5, 4.1))
plt.plot(freq_gaussian, np.abs(fft_gaussian), color ='b')
plt.xlabel('frequency')
plt.ylabel('FFT Magnitude')
#plt.savefig('5', dpi = 150)
plt.show()

ifft_gaussian = general_ifft(fft_gaussian, len(original_gaussian))

plt.figure(figsize = (9.5, 4.1))
plt.plot(time, ifft_gaussian, color ='b')
plt.xlabel('x')
plt.ylabel('y')
#plt.savefig('6', dpi = 150)
plt.show()


"PART 2"

path = '/Users/lysi2/Documents/UNI_Caltech/Ph_21/A2/arecibo1.txt'

data = np.array([np.float64(line) for line in open(path, "r")])



#plot data


plt.figure(figsize = (9.5, 4.1))
plt.plot(data, color ='b')
plt.ylabel('Data')
plt.xlabel('Index ')
#plt.savefig('7', dpi = 150)
plt.show()

data_fft , data_freq = general_fft(data, d = 0.001)

#calculate the max freq


freq_at_max_H = data_freq[np.argmax(np.abs(data_fft))]


print(freq_at_max_H)

#plot fourier

plt.figure(figsize = (9.5, 4.1))
plt.plot(data_freq, np.abs(data_fft), color ='b')
plt.ylabel('FFT Magnitude')
plt.xlabel('frequency ')
#plt.savefig('8', dpi = 150)
plt.show()


#zoom in


plt.figure(figsize = (9.5, 4.1))
plt.plot(data_freq, np.abs(data_fft), color ='b')
plt.ylabel('FFT Magnitude')
plt.xlabel('frequency ')
plt.xlim(134, 139)
#plt.savefig('9', dpi = 150)
plt.show()


#create gaussian and sin

def gaussian_envelope(t,t0, dt):
    gaussian_env = np.exp((-(t - t0) ** 2) / (2 * dt) ** 2)
    return gaussian_env

def perfect_sin(t,f):
    perf_sin = np.sin(2 * np.pi * f * t)
    return perf_sin


x_values = np.arange(-10, 10, step=.001)

#create convolutios with different dt

dt =0.02

conv = gaussian_envelope(x_values, 0, dt) * perfect_sin(x_values, freq_at_max_H)

dt_2 = 0.2

conv_2 = gaussian_envelope(x_values, 0, dt_2) * perfect_sin(x_values, freq_at_max_H)


dt_3 = 1

conv_3 = gaussian_envelope(x_values, 0, dt_3) * perfect_sin(x_values, freq_at_max_H)


#plot original convolution
plt.figure(figsize = (9.5, 4.1))
plt.plot(x_values, conv, color ='b')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-0.1, 0.1)
#plt.savefig('10', dpi = 150)
plt.show()


# find their fourier transforms

conv_fft, conv_freq = general_fft(conv, d= 0.001)

conv_fft2, conv_freq2 = general_fft(conv_2, d= 0.001)

conv_fft3, conv_freq3 = general_fft(conv_3, d= 0.001)


plt.figure(figsize = (9.5, 4.1))
plt.plot(conv_freq, np.abs(conv_fft), color ='b')
plt.xlabel('frequency')
plt.ylabel('FFT Magnitude')
#plt.savefig('11', dpi = 150)
plt.show()


#normalization in order to have same magnitude

max_magnitude = np.max(np.abs(data_fft))

#plot everything together

plt.figure(figsize = (9.5, 4.1))
plt.plot(conv_freq, np.abs(conv_fft) * (max_magnitude / np.max(np.abs(conv_fft))), color ='r', label = 'dt =0.02')
plt.plot(conv_freq2, np.abs(conv_fft2) * (max_magnitude / np.max(np.abs(conv_fft2))), color ='y', label = 'dt =0.2')
plt.plot(conv_freq3, np.abs(conv_fft3) * (max_magnitude / np.max(np.abs(conv_fft3))), color ='g', label = 'dt =1')
plt.plot(data_freq, np.abs(data_fft), color ='b', label ='data')
plt.ylabel('FFT Magnitude')
plt.xlabel('frequency ')
plt.legend()
plt.xlim(135.5, 138)
#plt.savefig('17', dpi = 150)
plt.show()

print(max_magnitude / np.max(np.abs(conv_fft)))

#Part 3

#use of lombscargle for gaussian

lomb_g_freq, lomb_g = LombScargle(time, original_gaussian, .01).autopower(minimum_frequency=0.01, maximum_frequency=1)

plt.figure(figsize = (9.5, 4.1))
plt.plot(lomb_g_freq, lomb_g, color ='b')
plt.xlabel('frequency')
plt.ylabel('lombscargle coefficients')
#plt.savefig('13', dpi = 150)
plt.show()

x_for_data = np.arange(len(data) * .001, step=.001)

lomb_data_freq, lomb_data = LombScargle(x_for_data, data, .01).autopower(minimum_frequency=0.1, maximum_frequency=500)


plt.figure(figsize = (9.5, 4.1))
plt.plot(lomb_data_freq, lomb_data, color ='b')
plt.xlabel('frequency')
plt.ylabel('lombscargle coefficients')
plt.xlim(0, 200)
#plt.savefig('14', dpi = 150)
plt.show()


freq_at_max_H = lomb_data_freq[np.argmax(np.abs(lomb_data))]

print(freq_at_max_H)

path2 = '/Users/lysi2/Documents/UNI_Caltech/Ph_21/A2/csvfile.csv'


data_Her_X = np.genfromtxt(path2, comments='#', delimiter=',', skip_header = 1)

magnitude  = data_Her_X[:,1]

mean_magnitude = np.mean(magnitude)

print(mean_magnitude)

MJD = data_Her_X[:,5]

dy = .0002
Her_lomb_freq, Her_lomb = LombScargle(MJD, magnitude - mean_magnitude, dy).autopower(minimum_frequency=.585, maximum_frequency=.59)


fig = plt.figure(figsize = (9.5, 4.1))
ax = fig.add_subplot(1, 1, 1)
# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 0.6, 0.0005)
minor_ticks = np.arange(0, 4, 0.5)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
#ax.set_yticks(major_ticks)
#ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')

plt.plot(Her_lomb_freq, Her_lomb, color ='b')
#plt.grid('True')
plt.xlabel('cycles/day')
plt.ylabel('lombscargle coefficients')
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.axvline(x=1/1.7, color ="r", linestyle = "--" )
#plt.savefig('21', dpi = 150)
plt.show()














