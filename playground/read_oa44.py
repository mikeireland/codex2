#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:29:37 2021

@author: mireland

To convert a CLV to a visibility curve use a Hankel transform. Faster than a
2D Fourier transform.
"""
#from astropy.table import TAble
import numpy as np
import matplotlib.pyplot as plt

file = 'OA44f.all1106.248480'
dd = np.loadtxt('OA44f.all1106.248480', skiprows=4)
f = open(file)
row1 = f.readline()
row2 = f.readline()
row3 = f.readline()
row4 = f.readline()
f.close()
radius_cm = float(row2.split()[-1])
sinth = np.array(row4.split()[3:]).astype(float)
wave = dd[:,0]
L = dd[:,1]
I0 = dd[:,2]
clvs = dd[:,3:]
Is = np.empty_like(clvs)
for i in range(I0.shape[0]):
    Is[i] = I0[i]*clvs[i]
x = np.arange(-5,6)
g = np.exp(-x**2/9)
g /= np.sum(g)

plt.figure(1)
plt.clf()
plt.plot(np.convolve(wave,g), np.convolve(L, g))
plt.axis([1.5,1.8,0,2e37])


#Convolve the intensity
plt.figure(2)
plt.clf()
I_conv = np.empty_like(Is)
for i in range(I_conv.shape[1]):
    I_conv[:,i] = np.convolve(Is[:,i], g, mode='same')
for i in range(6000,6010):
    plt.plot(sinth, I_conv[i])