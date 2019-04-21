# -*- coding: utf-8 -*-
import numpy as np;
from PIL import Image;
from scipy.ndimage import filters;
from PIL import ImageTk;
import cv2 as cv
import time;
#from Tkinter import *

def take_filter():
    """ give filter show of saved images """
    for i in range(1,11):
        im   = np.array(Image.open('img/screenshot'+ str(10*i) +'.png').convert('L'));
        imx  = np.zeros(im.shape);filters.sobel(im,1,imx);
        imy  = np.zeros(im.shape);filters.sobel(im,0,imy);
        imxy = np.sqrt(imx**2+imy**2);
        Image.fromarray( np.append(np.append(imx,imy,axis=1),imxy,axis=1)  ).convert('RGB').save('img_filter/f'+str(10*i)+'.jpg');
        cv.imshow('time',cv.imread('img_filter/f'+str(10*i)+'.jpg'));cv.waitKey(52);
        #time.sleep(1);
        #cv.destroyAllWindows();



if __name__ == '__main__':
    #res = [];
    #for i in range(1,50):res.append( cv.imread('img/screenshot'+str(10*i)+'.png')  );
    #i=5
    #for i in range(1,70):
    #	cv.imshow('time',cv.imread('img/screenshot'+str(10*i)+'.png'));cv.waitKey(52);
    #cv.imshow('time:',cv.imread('img_filter/f10.jpg'));
    #tk = Tk();
    #img_gif = tk.PhotoImage('img_filter/f10.jpg');
    #label_img = tk.Label(root, image = img_gif);
    #label_img.pack()
    take_filter();