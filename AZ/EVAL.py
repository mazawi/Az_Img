import cv2
import numpy as np


def precision_recall (img1, img2):
    
    h= img1.shape[0]
    w= img1.shape[1]
    
    ar1 = mask(img1)
    ar1 = ar1/255
    ar2 = mask(img2)
    ar2 = ar2/255
    draw_image (ar1)
    draw_image (ar2)
    s = 0
    sp = 0
    sh = 0
    for y in range (0,h):
        for x in range (0, w):
            s = s+ ar1[y,x]*ar2[y,x]
            sp += ar1[y,x]
            sh += ar2[y,x]
            
    p = s/sp
    r = s/sh
    f = (2*p*r)/(p+r)
    
    return p, r, f
######################################################################

def XOR (img1, img2):
    
    h= img1.shape[0]
    w= img1.shape[1]
    
    ar1 = mask(img1)
    ar1 = ar1/255
    ar2 = mask(img2)
    ar1 = ar1.astype(int)
    ar2 = ar2/255
    ar2 = ar2.astype(int)
    draw_image (ar1)
    draw_image (ar2)
    s = 0
    sp = 0
    sh = 0
    for y in range (0,h):
        for x in range (0, w):
            s = s+ (1- (ar1[y,x]^ar2[y,x]))
            
    s = s/(h*w)
    return s