#!/usr/bin/python 
# coding:utf-8 

import matplotlib.pyplot as plt; 
import tensorflow as tf;
import numpy as np;
from PIL import Image,ImageDraw;
import cv2;
import os;
from keras import metrics;
from keras.datasets import mnist
from keras.layers import Layer, Lambda, Subtract, Input, Dense, Reshape, Flatten, Dropout, Multiply, Dot, GlobalAveragePooling2D, Conv2DTranspose, Add;
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras import optimizers;
from keras import losses;
from keras import metrics;
from scipy import misc;
import glob;
from keras.callbacks import Callback;
import imageio;


# ============= make dataset ==================
# color     = ['black','red','green','blue','purple'];
# X_in_Arr  = [];
# X_out_Arr = []; 
# L_Arr     = [];
# dat_size  = 5000;
# img_dim   = 96;
# for i in range(dat_size):
#     num        = np.random.randint(low=0,high=1200);
#     col_in     = np.random.randint(low=0,high=5);
#     col_out    = np.random.randint(low=0,high=5);
#     X_in       = 'vae_data/bench_' + color[col_in] + '_' + str(num) + '.png';
#     X_out      = 'vae_data/bench_' + color[col_out] + '_' + str(num) + '.png';
#     X_in_Arr.append( misc.imresize(misc.imread(X_in),(img_dim, img_dim)).astype(np.float32)[:,:,0:-1] / 255 * 2 - 1 );
#     X_out_Arr.append( misc.imresize(misc.imread(X_out),(img_dim, img_dim)).astype(np.float32)[:,:,0:-1] / 255 * 2 - 1 );
#     L          = np.zeros(5);
#     L[col_out] = 1;
#     L_Arr.append(L);    
# np.save('X_in',X_in_Arr);
# np.save('X_out',X_out_Arr);
# np.save('Lan',L_Arr);

# ============= define model ==================
img_dim = 96;
z_dim   = 512;
x_in    = Input(shape=(img_dim, img_dim, 3))
x_right = Input(shape=(img_dim, img_dim, 3))
x       = x_in
x       = Conv2D(z_dim/16, kernel_size=(5,5), strides=(2,2), padding='SAME',trainable=False)(x)
x       = BatchNormalization(trainable=False)(x)
x       = LeakyReLU(0.2,trainable=False)(x)
x       = Conv2D(z_dim/8, kernel_size=(5,5), strides=(2,2), padding='SAME',trainable=False)(x)
x       = BatchNormalization(trainable=False)(x)
x       = LeakyReLU(0.2,trainable=False)(x)
x       = Conv2D(z_dim/4, kernel_size=(5,5), strides=(2,2), padding='SAME',trainable=False)(x)
x       = BatchNormalization(trainable=False)(x)
x       = LeakyReLU(0.2,trainable=False)(x)
x       = Conv2D(z_dim/2, kernel_size=(5,5), strides=(2,2), padding='SAME',trainable=False)(x)
x       = BatchNormalization(trainable=False)(x)
x       = LeakyReLU(0.2,trainable=False)(x)
x       = Conv2D(z_dim, kernel_size=(5,5), strides=(2,2), padding='SAME',trainable=False)(x)
x       = BatchNormalization(trainable=False)(x)
x       = LeakyReLU(0.2,trainable=False)(x)
x_en    = GlobalAveragePooling2D(trainable=False)(x)

encoder = Model(x_in, x_en)
encoder.summary();
map_size = K.int_shape(encoder.layers[-2].output)[1:-1]


z_in  = Input(shape=K.int_shape(x_en)[1:]);
z_h   = z_in;
l_in  = Input(shape=(5,));
l     = l_in;
z_h   = Dense(z_dim, activation='relu')(z_h);
l     = Dense(z_dim/4, activation='relu')(l);
l     = Dense(z_dim/2, activation='relu')(l);
l     = Dense(z_dim, activation='relu')(l);
lz    = Multiply(name='lz')([z_h,l]);
d_z   = LeakyReLU(0.2)(lz);
z_en = Add()([z_in,d_z]);

hidden = Model([z_in,l_in],z_en);
hidden.summary();


de_in = Input(shape=(z_dim,));
z = de_in;
z = Dense(np.prod(map_size)*z_dim,trainable=False)(z)
z = Reshape(map_size + (z_dim,),trainable=False)(z)
z = Conv2DTranspose(z_dim/2, kernel_size=(5,5), strides=(2,2), padding='SAME',trainable=False)(z)
z = BatchNormalization(trainable=False)(z)
z = Activation('relu',trainable=False)(z)
z = Conv2DTranspose(z_dim/4, kernel_size=(5,5), strides=(2,2), padding='SAME',trainable=False)(z)
z = BatchNormalization(trainable=False)(z)
z = Activation('relu',trainable=False)(z)
z = Conv2DTranspose(z_dim/8, kernel_size=(5,5), strides=(2,2), padding='SAME',trainable=False)(z)
z = BatchNormalization(trainable=False)(z)
z = Activation('relu',trainable=False)(z)
z = Conv2DTranspose(z_dim/16, kernel_size=(5,5), strides=(2,2), padding='SAME',trainable=False)(z)
z = BatchNormalization(trainable=False)(z)
z = Activation('relu',trainable=False)(z)
z = Conv2DTranspose(3, kernel_size=(5,5), strides=(2,2), padding='SAME',trainable=False)(z)
z = Activation('tanh',trainable=False)(z)

decoder = Model(de_in, z)
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



z_shift     = Dense(z_dim)(x_en);
z_log_scale = Dense(z_dim)(x_en);
u = Lambda(lambda z: K.random_normal(shape=K.shape(z)))(z_shift)
z = ScaleShift()([u, z_shift, z_log_scale])

z = hidden([z,l_in])

x_recon = decoder(z)
x_out = Subtract()([x_right, x_recon])

recon_loss = 0.5 * K.sum(K.mean(x_out**2, 0)) + 0.5 * np.log(2*np.pi) * np.prod(K.int_shape(x_out)[1:])
z_loss = 0.5 * K.sum(K.mean(z**2, 0)) - 0.5 * K.sum(K.mean(u**2, 0))
vae_loss = recon_loss + z_loss;
se_vae = Model(inputs=[x_in,x_right,l_in],outputs=x_out);
se_vae.add_loss(vae_loss);
se_vae.compile(optimizer=Adam(1e-4));

# ============= read data ===============
X_in  = np.load('X_in.npy');
X_out = np.load('X_out.npy');
L_lan = np.load('Lan.npy');



def data_generator(batch_size=16):
    X1 = [];X2 = [];Y = [];
    while True:
        for i in range(batch_size):
            data_id = np.random.randint(low=0,high=len(X_in));
            x_in  = X_in[ data_id ];
            x_out = X_out[ data_id ];
            l_lan = L_lan[ data_id ];
            X1.append(x_in);
            X2.append(l_lan);
            Y.append(x_out);
            if len(X1) == batch_size:
                X1 = np.array(X1);
                X2 = np.array(X2);
                Y  = np.array(Y);
                #yield ({'input_1': X1, 'input_3': X2}, {'output': Y})#,None
                yield ({'input_1': X1, 'input_3': X2})#,None
                X1 = [];X2 = [];Y = [];

# ============= train ===================
def sample(path):
    n = 3
    figure = np.zeros((img_dim*n, img_dim*n, 3))
    for i in range(n):
        for j in range(n):
            x_recon = decoder.predict(np.random.randn(1, z_dim));
            #data_id  = np.random.randint(low=0,high=len(X_in));
            #x_en     = encoder.predict(np.array([ X_in[data_id] ]) );
            #z_hidden = hidden.predict(np.array([  x_en[0],L_lan[data_id]  ]) );
            #x_recon  = decoder.predict(z_hidden);
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
        path = 'samples_se/test_%s.png' % epoch
        sample(path)
        self.losses.append((epoch, logs['loss']))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('./best_encoder.weights')
            hidden.save_weights('./best_hidden.weights')
            decoder.save_weights('./best_decoder.weights')

evaluator = Evaluate()

# se_vae.fit_generator(data_generator(),
#                   epochs=100,
#                   steps_per_epoch=100,
#                   callbacks=[evaluator])

#se_vae.fit(x=[X_in,L_lan],y=[X_out],epochs=100, batch_size=10);
encoder.load_weights('best_encoder.weights');
decoder.load_weights('best_decoder.weights');
se_vae.fit([X_in,X_out,L_lan],epochs=100, batch_size=10, callbacks=[evaluator]);