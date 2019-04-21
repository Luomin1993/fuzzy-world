#!/usr/bin/python 
# coding:utf-8 

import matplotlib.pyplot as plt; 
import tensorflow as tf;
import numpy as np;
from PIL import Image,ImageDraw;
import cv2;
import os;

g_objs = ['pipe_normal_','chair_high_','truck_normal_','horse_normal_','music_normal_'];
objs          = ['pipe','chair','truck','horse','music'];
relationships = ['up_left','up_right','down_left','down_right','left','right','down','up'];
BBoxes = np.load('data_bboxes.npy');
Objs   = np.load('data_objs.npy');
States = np.load('data_states.npy');
for index in range(20):
    img = np.array(Image.open('scene/sc' +str(index)+ '.png'));
    #img = np.zeros((512,512,3), dtype = np.uint8)
    for i_bbox in range(len(BBoxes[index])):
        w = BBoxes[index][i_bbox][2];
        h = BBoxes[index][i_bbox][3];
        # cv2.line(img, (bbox[0],bbox[1]), (bbox[0]+w,bbox[1]), (0, 255,0),1);
        # cv2.line(img, (bbox[0],bbox[1]), (bbox[0],bbox[1]+h), (0, 255,0),1);
        # cv2.line(img, (bbox[0]+w,bbox[1]), (bbox[0]+w,bbox[1]+h), (0, 255,0),1);
        # cv2.line(img, (bbox[0],bbox[1]+h), (bbox[0]+w,bbox[1]+h), (0, 255,0),1);
        cv2.rectangle(img,(BBoxes[index][i_bbox][0],BBoxes[index][i_bbox][1]), (BBoxes[index][i_bbox][0]+w,BBoxes[index][i_bbox][1]+h),(255,0,0),2);
        cv2.putText(img,objs[Objs[index][i_bbox]],(BBoxes[index][i_bbox][0],BBoxes[index][i_bbox][1]),cv2.FONT_HERSHEY_PLAIN,1.4,(111,111,255),1);
    #Image.fromarray(img).show();
    for rel in States[index]:
        w1     = BBoxes[index][Objs[index].index(rel[1])][2];
        h1     = BBoxes[index][Objs[index].index(rel[1])][3];
        w2     = BBoxes[index][Objs[index].index(rel[2])][2];
        h2     = BBoxes[index][Objs[index].index(rel[2])][3];
        point1 = ( BBoxes[index][Objs[index].index(rel[1])][0]+w1/2 ,BBoxes[index][Objs[index].index(rel[1])][1]+h1/2);
        point2 = ( BBoxes[index][Objs[index].index(rel[2])][0]+w2/2 ,BBoxes[index][Objs[index].index(rel[2])][1]+h2/2);
        rel    = relationships[rel[0]];
        cv2.line(img, point1, point2, (0, 255,0),1);
        cv2.putText(img,rel,(  (point1[0]+point2[0])/2,  (point1[1]+point2[1])/2),cv2.FONT_HERSHEY_PLAIN,1.4,(222,222,11),1);
    Image.fromarray(img).save('scene_box/box_sc'+str(index)+'.png');

# ============ make data ===============
# img_names = os.listdir('./scene');
# # i=0;total=len(img_names)
# IMGs = [];
# for img_name in img_names:
#     IMGs.append( np.array(Image.open('./scene/'+img_name).convert('L').resize((96,96))) );
# np.save('scene_img',np.array(IMGs));


# ============ Model ===================
# from keras.datasets import mnist
# from keras.layers import Input, Dense, Reshape, Flatten, Dropout
# from keras.layers import BatchNormalization, Activation, ZeroPadding2D
# from keras.layers.advanced_activations import LeakyReLU
# from keras.layers.convolutional import UpSampling2D, Conv2D
# from keras.models import Sequential, Model
# from keras.optimizers import Adam
# from keras import backend as K
# from keras import optimizers;
# from keras import losses;


# X_input =Input(shape=(96,96,1));
# X = Conv2D(32, kernel_size=3, strides=2, input_shape=(96,96,1), padding="same")(X_input);
# X = LeakyReLU(alpha=0.2)(X);
# X = Dropout(0.25)(X);
# X = Conv2D(64, kernel_size=3, strides=2, padding="same")(X);
# X = ZeroPadding2D(padding=((0,1),(0,1)))(X);
# X = BatchNormalization(momentum=0.8)(X);
# X = LeakyReLU(alpha=0.2)(X);
# X = Dropout(0.25)(X);
# X = Conv2D(128, kernel_size=3, strides=2, padding="same")(X);
# X = BatchNormalization(momentum=0.8)(X);
# X = LeakyReLU(alpha=0.2)(X);
# X = Dropout(0.25)(X);
# X = Conv2D(256, kernel_size=3, strides=1, padding="same")(X);
# X = BatchNormalization(momentum=0.8)(X);
# X = LeakyReLU(alpha=0.2)(X);
# X = Dropout(0.25)(X);
# X = Flatten()(X);
# X = Dense(256, activation='sigmoid')(X);
# X = Dense(512, activation='sigmoid')(X);
# Y = Dense(32*32, activation='sigmoid')(X);


# model  = Model(inputs=X_input, outputs=Y);
# sgd    = optimizers.SGD(lr=0.00001, decay=0.0, momentum=0.9, nesterov=True);
# model.compile(optimizer=sgd, loss=losses.mean_squared_error, metrics=['accuracy']);

# # =========== read data =================
# X_img = np.load('scene_img.npy')
# X_img = np.expand_dims(X_img, axis=3);
# Y_box = np.load('data_bboxes_mat.npy')
# Y_box = np.reshape(Y_box,(len(Y_box),32*32));
# # print X_img.shape;
# # print Y_box.shape;


# # =========== train model ==============
# model.fit([X_img], [Y_box],epochs=12, batch_size=1);
# model.save('model_dete.h5');


# ============ use model ===========
# from keras.models import Model, load_model;
# model_dete = load_model('model_dete.h5');
# test_img   = np.array(Image.open('scene/sc7.png').convert('L').resize((96,96)));
# test_img   = np.expand_dims(test_img, axis=3);
# print model_dete.predict(np.array([test_img]));