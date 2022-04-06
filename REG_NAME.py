import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def register_images(path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    i =0
    for img1 in os.listdir(path):
        img_path = os.path.join(path, img1)
        img = cv2.imread(img_path)
        path_test = img_path.split('.')
        if path_test[len(path_test)-1] in ['jpg', 'jpeg']:
            print(img_path)
            if i<10: 
                fn = '00{}.jpg'.format(i)
            elif i<100 and i>=10:
                fn = '0{}.jpg'.format(i)
            else:
                fn = '{}.jpg'.format(i)

            filename = save_path+'/' +fn

            cv2.imwrite(filename, img)
            i +=1
