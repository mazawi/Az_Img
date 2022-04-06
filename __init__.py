import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

def wait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_image_info(img):
    image_height = img.shape[0]
    image_width = img.shape[1]
    image_ch = img.shape[2]
    return  image_height,image_width,image_ch


    
def draw_image (img, title='Az draw image, replace this title'):
    plt.imshow(img)
    plt.title(title)
    plt.show()

def draw_gray_image (img, title='title'):
    plt.imshow(img, cmap='gray' )
    plt.title(title)
    plt.show()
    
def resize_image(img,w =200, h=200):
    rimg = cv2.resize(img, (w,h))
    return rimg
    
def correct_color (img):
    rimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rimg

def crop_image(in_image, x1, x2, y1, y2):
    rimg = in_image[y1:y2, x1:x2]
    return rimg

# def crop_image(img):
#     rimg1 = img[0:200, 0:100]
#     rimg2 = img[0:200, 100:200]
#     return rimg1, rimg2

def gray_image(img):
    rimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return rimg

def image_binarization(img, type = 'mid', th = 127):
    if type =='mid':
        ret, rimg = cv2.threshold(img, th,255, cv2.THRESH_BINARY)
    elif type == 'adaptive':
        rimg = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,115,1)
    elif type == 'otsu':
         ret, rimg = cv2.threshold(img, th,255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)  
    
    return  rimg
    
def gabor (im, ksize=5, sigma=3, theta=1*np.pi/4, lamda=1*np.pi/4,gamma=0.5, phi=0, ktype=cv2.CV_32F ):
    kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype)
    rimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    return rimg

def edge(img, type = 'canny', low = 150, high = 200, size = 3):
    if type == 'canny' :
        rimg = cv2.Canny(img, low, high, size)
    elif type == 'lap':
        rimg = cv2.Laplacian(img,cv2.CV_64F)
    elif type == 'sob':
        rimgx = cv2.Sobel(img, cv2.CV_64F, 1, 0, 3)
        rimgy = cv2.Sobel(img, cv2.CV_64F, 0, 1, 3)
        rimg = cv2.bitwise_or(rimgx, rimgy)
    return rimg



def lines (img, dist=1, theta= np.pi/180, thres=200):
    lin = cv2.HoughLines(img, dist, theta, thres)
    return lin

def read_image(path):
    img = cv2.imread(path)
    return img

