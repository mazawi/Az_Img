import cv2
import numpy as np

######################################################################
def cluster_image(img, K=2):
    
    #img = image_gray(img)
        
    Z = img.reshape((-1,3)) 
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    return res2

#####################################################################

def binarize_image(img, type = 'mid', th = 127):
    if type =='mid':
        ret, rimg = cv2.threshold(img, th,255, cv2.THRESH_BINARY)
    elif type == 'adaptive':
        rimg = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,115,1)
    elif type == 'otsu':
         ret, rimg = cv2.threshold(img, th,255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU) 
    return rimg  


#############################################################################
def canny(img, low = 150, high = 200, size = 3):
    rimg = cv2.Canny(img, low, high, size)
    return rimg

###############################################################################
def laplacian(img):
    rimg = cv2.Laplacian(img,cv2.CV_64F)

    return rimg

#############################################################################
def sobelx(img,ksize = 5):
    rimg = cv2.Sobel(img,cv2.CV_64F,1,0,ksize)
    return rimg

#############################################################################
def sobely(img,ksize = 5):
    rimg = cv2.Sobel(img,cv2.CV_64F,0,1,ksize)
    return rimg

#############################################################################
def sobel (img,ksize = 5):
    rimgx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize)
    rimgy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize)
    rimg = cv2.bitwise_or(rimgx, rimgy)
    return  rimg


###############################################################################

def find_borders (in_image):
    h1 = in_image.shape[0]
    w1 = in_image.shape[1]
    th1 = 255*8

    for i in range(0,h1):
        s = np.sum(in_image[i,:])
        if s > th1:
            Y1 = i
            break

    for i in range(h1-1,0,-1):
        s = np.sum(in_image[i,:])
        if s > th1:
            Y2 = i
            break


    for i in range(0,w1):
        s = np.sum(in_image[:,i])
        if s > th1:
            X1 = i
            break

    for i in range(w1-1,0,-1):
        s = np.sum(in_image[:,i])
        if s > th1:
            X2 = i
            break
    return X1, X2, Y1, Y2

###################################################################

def crop_image(in_image, x1, x2, y1, y2):
    rimg = in_image[y1:y2, x1:x2]
    return rimg

##################################################################

def remove_borders(imgi):
    
    msk = mask (imgi)
    x1, x2, y1, y2 = find_borders (msk)
    img = crop_image(imgi,x1, x2, y1, y2)
    
    return img