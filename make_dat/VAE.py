#!/usr/bin/python 
# coding:utf-8 

import matplotlib.pyplot as plt; 
import tensorflow as tf;
import numpy as np;
from PIL import Image,ImageDraw;
import cv2;
import os;
from keras import metrics

# g_objs = ['pipe_normal_','chair_high_','truck_normal_','horse_normal_','music_normal_'];
# objs          = ['pipe','chair','truck','horse','music'];
# relationships = ['up_left','up_right','down_left','down_right','left','right','down','up'];
# BBoxes = np.load('data_bboxes.npy');
# Objs   = np.load('data_objs.npy');
# States = np.load('data_states.npy');
# for index in range(20):
#     img = np.array(Image.open('scene/sc' +str(index)+ '.png'));
#     #img = np.zeros((512,512,3), dtype = np.uint8)
#     for i_bbox in range(len(BBoxes[index])):
#         w = BBoxes[index][i_bbox][2];
#         h = BBoxes[index][i_bbox][3];
#         # cv2.line(img, (bbox[0],bbox[1]), (bbox[0]+w,bbox[1]), (0, 255,0),1);
#         # cv2.line(img, (bbox[0],bbox[1]), (bbox[0],bbox[1]+h), (0, 255,0),1);
#         # cv2.line(img, (bbox[0]+w,bbox[1]), (bbox[0]+w,bbox[1]+h), (0, 255,0),1);
#         # cv2.line(img, (bbox[0],bbox[1]+h), (bbox[0]+w,bbox[1]+h), (0, 255,0),1);
#         cv2.rectangle(img,(BBoxes[index][i_bbox][0],BBoxes[index][i_bbox][1]), (BBoxes[index][i_bbox][0]+w,BBoxes[index][i_bbox][1]+h),(255,0,0),2);
#         cv2.putText(img,objs[Objs[index][i_bbox]],(BBoxes[index][i_bbox][0],BBoxes[index][i_bbox][1]),cv2.FONT_HERSHEY_PLAIN,1.4,(111,111,255),1);
#     #Image.fromarray(img).show();
#     for rel in States[index]:
#         w1     = BBoxes[index][Objs[index].index(rel[1])][2];
#         h1     = BBoxes[index][Objs[index].index(rel[1])][3];
#         w2     = BBoxes[index][Objs[index].index(rel[2])][2];
#         h2     = BBoxes[index][Objs[index].index(rel[2])][3];
#         point1 = ( BBoxes[index][Objs[index].index(rel[1])][0]+w1/2 ,BBoxes[index][Objs[index].index(rel[1])][1]+h1/2);
#         point2 = ( BBoxes[index][Objs[index].index(rel[2])][0]+w2/2 ,BBoxes[index][Objs[index].index(rel[2])][1]+h2/2);
#         rel    = relationships[rel[0]];
#         cv2.line(img, point1, point2, (0, 255,0),1);
#         cv2.putText(img,rel,(  (point1[0]+point2[0])/2,  (point1[1]+point2[1])/2),cv2.FONT_HERSHEY_PLAIN,1.4,(222,222,11),1);
#     Image.fromarray(img).save('scene_box/box_sc'+str(index)+'.png');

# ============ make data ===============
# def e2f(A_semantic_data):
#     res = np.zeros((len(A_semantic_data),4));
#     for i in range(len(A_semantic_data)):
#         if A_semantic_data[i][0]==1. or A_semantic_data[i][3]==1.:res[i][0]=1.;
#         if A_semantic_data[i][1]==1. or A_semantic_data[i][2]==1.:res[i][1]=1.;
#         if A_semantic_data[i][4]==1. or A_semantic_data[i][5]==1.:res[i][2]=1.;
#         if A_semantic_data[i][6]==1. or A_semantic_data[i][7]==1.:res[i][3]=1.;
#     return res;    

# img_names = os.listdir('./img');
# IMGs = [];
# for img_name in img_names:
#     IMGs.append( np.array(Image.open('./img/'+img_name).resize((96,96))) );
# np.save('IMGs',np.array(IMGs)[:,:,:,0:-1]);

# for i in range(len(IMGs)):
#     # print len(data_des[i])==len(data_que[i])
#     for j in range(len(data_que[i])):
#         visual_data.append(IMGs[i]);
#         Q_semantic_data.append(data_que[i][j]);
#         A_semantic_data.append(data_des[i][j]);
# np.save('X_img_i',np.array(visual_data));
# np.save('Q_s',np.array(Q_semantic_data));
# np.save('A_s',np.array(A_semantic_data));

# C_semantic_data = [];
# A_t             = [];
# data_cmd        = np.load('data_cmd.npy');
# action          = np.load('data_act.npy');
# for i in range(len(IMGs)):
#     print len(data_cmd[i])==len(action[i]);
#     for j in range(len(data_cmd[i])):
#         visual_data.append(IMGs[i]);
#         C_semantic_data.append(data_cmd[i][j]);
#         A_t.append(action[i][j])
# np.save('X_img_g',np.array(visual_data));
# np.save('C_semantic_data',np.array(C_semantic_data));
# np.save('Y_G',np.array(A_t));


# # ============ eval ===================
# def mean_pred(y_true, y_pred):
#     return K.mean(y_pred)

# ============ Model ===================
from keras.datasets import mnist
from keras.layers import Layer, Lambda, Subtract, Input, Dense, Reshape, Flatten, Dropout, Multiply, Dot, GlobalAveragePooling2D, Conv2DTranspose;
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras import optimizers;
from keras import losses;
from keras import metrics;
from scipy import misc
import glob
from keras.callbacks import Callback;
import imageio;

#-------------------  rec ----------------------
img_dim = 96;
z_dim = 512;
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in
x = Conv2D(z_dim/16, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim/8, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim/4, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim/2, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = GlobalAveragePooling2D()(x)

encoder = Model(x_in, x)
encoder.summary()
map_size = K.int_shape(encoder.layers[-2].output)[1:-1]

# 解码层，也就是生成器部分
z_in = Input(shape=K.int_shape(x)[1:])
z = z_in
z = Dense(np.prod(map_size)*z_dim)(z)
z = Reshape(map_size + (z_dim,))(z)
z = Conv2DTranspose(z_dim/2, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(z_dim/4, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(z_dim/8, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(z_dim/16, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(3, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = Activation('tanh')(z)

decoder = Model(z_in, z)
decoder.summary()

class ScaleShift(Layer):
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def call(self, inputs):
        z, shift, log_scale = inputs
        z = K.exp(log_scale) * z + shift
        logdet = -K.sum(K.mean(log_scale, 0))
        self.add_loss(logdet)
        return z

z_shift = Dense(z_dim)(x)
z_log_scale = Dense(z_dim)(x)
u = Lambda(lambda z: K.random_normal(shape=K.shape(z)))(z_shift)
z = ScaleShift()([u, z_shift, z_log_scale])

x_recon = decoder(z)
x_out = Subtract()([x_in, x_recon])

recon_loss = 0.5 * K.sum(K.mean(x_out**2, 0)) + 0.5 * np.log(2*np.pi) * np.prod(K.int_shape(x_out)[1:])
z_loss = 0.5 * K.sum(K.mean(z**2, 0)) - 0.5 * K.sum(K.mean(u**2, 0))
vae_loss = recon_loss + z_loss

vae = Model(x_in, x_out)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(1e-4))


# =========== read data =================
# X_img = np.load('IMGs.npy');
imgs = glob.glob('vae_data/*.png')
np.random.shuffle(imgs)

height,width = misc.imread(imgs[0]).shape[:2]
center_height = int((height - width) / 2)

def imread(f):
    x = misc.imread(f)
    x = x[center_height:center_height+width, :]
    x = misc.imresize(x, (img_dim, img_dim))
    return x.astype(np.float32) / 255 * 2 - 1

def data_generator(batch_size=16):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)[:,:,:,0:-1];
                yield X,None
                X = []


# =========== train models ==============
def sample(path):
    n = 9
    figure = np.zeros((img_dim*n, img_dim*n, 3))
    for i in range(n):
        for j in range(n):
            x_recon = decoder.predict(np.random.randn(1, *K.int_shape(x)[1:]))
            digit = x_recon[0]
            figure[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    figure = (figure + 1) / 2 * 255
    imageio.imwrite(path, figure)


class Evaluate(Callback):
    def __init__(self):
        import os
        self.lowest = 1e10
        self.losses = []
        if not os.path.exists('samples'):
            os.mkdir('samples')
    def on_epoch_end(self, epoch, logs=None):
        path = 'samples/test_%s.png' % epoch
        sample(path)
        self.losses.append((epoch, logs['loss']))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('./best_encoder.weights')
            decoder.save_weights('./best_decoder.weights')

evaluator = Evaluate()

vae.fit_generator(data_generator(),
                  epochs=100,
                  steps_per_epoch=100,
                  callbacks=[evaluator])

# # ============ use model ===========
# from keras.models import Model, load_model;
# model_dete = load_model('model_reco.h5');
# test_img   = np.array(Image.open('scene/sc117.png').convert('L').resize((96,96)));
# test_img   = np.expand_dims(test_img, axis=3);
# test_question = np.array([[1.,0,1.,0,0]]);
# # test_question   = np.expand_dims(test_question, axis=2);
# print model_dete.predict([np.array([test_img]),test_question]);