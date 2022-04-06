import cv2
import numpy as np

#####################################################################

def binarize_image(img, type = 'mid', th = 127):
    if type =='mid':
        ret, rimg = cv2.threshold(img, th,255, cv2.THRESH_BINARY)
    elif type == 'adaptive':
        rimg = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,115,1)
    elif type == 'otsu':
         ret, rimg = cv2.threshold(img, th,255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU) 
    return rimg  


###########################################################################
            
def resize_image(img,w =200, h=200):
    rimg = cv2.resize(img, (w,h))
    return rimg

################################################################################

def maskOTSU  (imgi):
    
    if(len(imgi.shape)>2):
        img = cv2.cvtColor(imgi, cv2.COLOR_BGR2GRAY)
    else:
        img = imgi
    h= img.shape[0]
    k_size =int ( h/10 +1)
    
    
    
    ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    kernel = np.ones((k_size,k_size),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return closing


##################################################################

def correct_mask(img, ofs=2): 
    x = img.shape[0]
    y = img.shape[1]
    
    img1 = resize_image (img, x-2*ofs,y-2*ofs)
    
    img2 = np.zeros((x,y))
    img2[ofs:x-ofs,ofs:y-ofs]= img1
    return img2

##########################################################
def masking (img1, img2):
    msr1 = np.array(img1)
    msr1 = msr1.astype(int)
    
    msr2 = np.array(img2)
    msr2 = msr2.astype(int)
    
    r = cv2.bitwise_and(msr1, msr2)
    return r

###############################################################################
def mask  (imgi):
    
    if(len(imgi.shape)>2):
        img = cv2.cvtColor(imgi, cv2.COLOR_BGR2GRAY)
    else:
        img = imgi
    h= img.shape[0]
    k_size =int ( h/10 +1)
    
    
    
    ret, thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
    kernel = np.ones((k_size,k_size),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return closing

     