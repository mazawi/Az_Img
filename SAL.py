import cv2
import numpy as np

#############################################################################
def irregAR(A):
    mean = np.mean(A)
    std = np.std(A)
    B = np.absolute((A-mean)*std)
    print ('mean, std = ', mean, std)
    
    return B

#############################################################################
def irregIM(img):
    
    if(len(img.shape)>2):
        rimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: 
        rimg = img 
        
    A = np.array(rimg)
    mean = np.mean(A)
    std = np.std(A)
    B = np.absolute((A-mean)*std)
    print ('mean, std = ', mean, std)
    
    return B


