# -*- coding: utf-8 -*-
import numpy as np;
import re;
import pyglet;
from pyglet.window import key;
import ratcave as rc;
from PIL import Image;
import cv2 as cv;
from scipy.ndimage import filters;
import matplotlib.pyplot as plt;
from matplotlib.patches import Circle;
from keras.models import Model, load_model;
from model_mine import *

# ========= define your task func here ===========
# def task_finish(y_true, y_pred):
#     return K.mean(K.greater(K.dot(K.softmax(y_pred), K.transpose(y_true)),.3), axis=-1)



# ========= read your dataset here ================
data_l      = np.load('DatGraph/DATA_CM.npy');
data_a      = np.load('DatGraph/DATA_ACT.npy');
data_q      = np.load('DatGraph/DATA_QV.npy');
data_o      = np.load('DatGraph/DATA_IMG.npy').astype('float32');
data_o      = np.reshape(data_o, (len(data_o), 64,64,1))
data_Gi     = np.load('DatGraph/G_I.npy');
data_Gi     = np.array([data_Gi for i in range(len(data_a))]);
data_Gs     = np.load('DatGraph/G_S.npy');
data_Gi     = np.reshape(data_Gi, (len(data_Gi), data_Gi.shape[1]*data_Gi.shape[-1]));
data_Gs     = np.reshape(data_Gs, (len(data_Gs), data_Gs.shape[1]*data_Gs.shape[2]*data_Gs.shape[-1]));
#data_a_last = np.array( [np.random.random(DIM_a).tolist()] + data_a[0:-1].tolist()  );  
data_a_last = data_a;data_a_last[1:]=data_a[0:-1];
data_Gs_    = data_Gs;data_Gs_[1:]=data_Gs[0:-1];
#print data_a_last[0];

#========= load model here ============
mine = load_model('mine.h5');