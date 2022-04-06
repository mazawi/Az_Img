import cv2


#####################################################################

def binarize_image(img, type = 'mid', th = 127):
    if type =='mid':
        ret, rimg = cv2.threshold(img, th,255, cv2.THRESH_BINARY)
    elif type == 'adaptive':
        rimg = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,115,1)
    elif type == 'otsu':
         ret, rimg = cv2.threshold(img, th,255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU) 
    return rimg  


##############################################################
def gray_image(imgi):
    
    if(len(imgi.shape)>2):
        rimg = cv2.cvtColor(imgi, cv2.COLOR_BGR2GRAY)
    else:
        rimg = imgi
    
    return rimg

#############################################################
def gray2RGB(imgi):
    
    if(len(imgi.shape)<3):
        rimg = cv2.cvtColor(imgi, cv2.COLOR_GRAY2RGB)
    else:
        rimg = imgi
    
    return rimg


##############################################################################

def HSV (img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv

##############################################################################
def image_gray(imgi):
    
    if(len(imgi.shape)>2):
        rimg = cv2.cvtColor(imgi, cv2.COLOR_BGR2GRAY)
    else:
        rimg = imgi
    
    return rimg