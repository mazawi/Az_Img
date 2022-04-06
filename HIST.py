import cv2
import numpy as np

def hist_calc(img):
    image_height = img.shape[0]
    image_width = img.shape[1]
    image_ch = img.shape[2]
    hist = np.zeros([256, image_ch], np.int32)

    for y in range(0,image_height):
        for x in range(0,image_width):
            for c in range(0,image_ch):
                hist[img[y,x,c],c] +=1

    return  hist
