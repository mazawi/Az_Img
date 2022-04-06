import cv2
from matplotlib import pyplot as plt

#################################################################################

def draw_image(img, title="Az Image show window"):
    plt.imshow(img,  cmap='gray')
    plt.title(title)
    plt.show()

#################################################################################

def read_image(path):
    try:
        imgr = cv2.imread(path)
        return imgr
    except:
        print('error reading image')
        return

#################################################################################
def save_image(img, name="noname.jpg"):
    cv2.imwrite(name, img)
    return
    
    
   
 #################################################################################    
def resize_image(img,w =200, h=200):
    rimg = cv2.resize(img, (w,h))
    return rimg

#################################################################################    
def mask (imgi):
    
    if(len(imgi.shape)>2):
        img = cv2.cvtColor(imgi, cv2.COLOR_BGR2GRAY)
    else:
        img = imgi
    h= img.shape[0]
    k_size =int ( h/10 +1)
    
    
    
    #ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
    kernel = np.ones((k_size,k_size),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return closing