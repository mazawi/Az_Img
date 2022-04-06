import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from skimage.filters import gabor
import pandas as pd
import os



def Dist (x, y):
    r = math.sqrt ((x-y)*(x-y))
    return r

###################################################################
# Histogram 
###################################################################
def Hist_features (img):
    img_array = np.array(img)
    hist = cv2.calcHist([img_array],[0],None,[256],[0,256])
    for i in range(0,10):
        hist[i]= 0
    H_mean = hist.mean()
    H_std = hist.std()
    
    
    return H_mean, H_std


###################################################################
# LBP 
###################################################################

def LBP(img):
    img_array = np.array(img)
    LBP = local_binary_pattern(img_array,8,1,'uniform')
    LBP = np.uint8((LBP/LBP.max())*255)

    LBP_hist, _ = np.histogram(LBP,8)
    LBP_hist = np.array(LBP_hist, dtype=float)
    prob = np.divide(LBP_hist , np.sum(LBP_hist ))
    E_energy = np.sum(prob**2)
    E_entropy = -np.sum(np.multiply(prob,np.log2(prob)))
    
    return E_energy, E_entropy

###################################################################
# GLCM 
###################################################################

def GLCM(img):
    img_array = np.array(img)
    glcm = greycomatrix (img_array, [1],[0],256,symmetric = True, normed = True)

    G_contrast = greycoprops(glcm, prop = 'contrast')
    G_dissimilarity = greycoprops(glcm, prop = 'dissimilarity')
    G_homogeneity = greycoprops(glcm, prop = 'homogeneity')
    G_energy = greycoprops(glcm, prop = 'energy')
    G_correlation = greycoprops(glcm, prop = 'correlation')

    return  G_energy[0][0], G_dissimilarity[0][0], G_homogeneity[0][0], G_contrast[0][0], G_correlation[0][0] 

###################################################################
# Gabor
###################################################################

def Gabor(img):
    img_array = np.array(img)
    gabor_r, gabor_j  = gabor(img_array, frequency= 0.6)
    gabor_g = (gabor_r**2 +gabor_j**2)//2

    gabor_hist, _ = np.histogram(gabor_g, 8)
    gabor_hist = np.array(gabor_hist, dtype = float)
    gabor_prob = np.divide(gabor_hist, np.sum(gabor_hist))
    gabor_en = np.sum(gabor_prob**2)
    gabor_ent = -np.sum(np.multiply (gabor_prob, np.log2(gabor_prob)))
    
    return gabor_en, gabor_ent

###################################################################
# power_spectrum
###################################################################

def power_spectrum(image):
     

    npix = image.shape[0]

    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    pl.loglog(kvals, Abins)
    pl.xlabel("$k$")
    pl.ylabel("$P(k)$")
    pl.tight_layout()
    pl.savefig("cloud_power_spectrum.png", dpi = 300, bbox_inches = "tight")
    
    return Abins

    