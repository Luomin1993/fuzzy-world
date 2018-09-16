# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:02:19 2017

@author: Hanss401
"""
import numpy as np;
import re;
from keras.layers import Input, Dense;
from keras.models import Model, load_model;
from keras.models import Sequential;
from keras.layers import Embedding;
from keras.layers import LSTM;
from keras.layers import Bidirectional;
from keras.layers import Dense;
from keras.layers import TimeDistributed;
from keras.layers import Dropout;
from keras.layers import Add;
from keras.layers.recurrent import GRU;
from keras import optimizers;
from keras import losses;

# ========= define your helper data here ==========
ACTION2WORDS = {};
WORDS_NUM = 33;
SEN_LEN   = 8;
IMG_DIM   = 64;
WORD_VEC_DIM = 14;
DIM_hm    = 72;
DIM_ha    = 72;
DIM_a     = 14;
DIM_f     = 72;

# ========= define model ============
l      = Input(shape=(WORDS_NUM,SEN_LEN), dtype='float32', name='l'); # inputed command;
o      = Input(shape=(IMG_DIM,IMG_DIM), dtype='float32', name='o'); # inputed image;
T      = LSTM((SEN_LEN,WORD_VEC_DIM), input_dim=WORDS_NUM, input_length=SEN_LEN,return_sequences=True)(l); # the output word-vec; 
C      = Conv2D(64, (5,5), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(o);
C      = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C);
C      = Conv2D(32, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(C);
C      = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C);
C      = Conv2D(16, (2,2), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(C); 
C      = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C);
C      = Flatten()(C); # the output cnn features; 
C_     = concatenate(axis=-1)(C,T); # the output cat;
C_     = Dense(144, activation='relu')(C_);
C_     = Dense(72, activation='relu')(C_); # the output GFT;
#h_m    = Embedding(72,input_length=1)(C_);
h_m    = GRU(units=DIM_hm)(C_);
a_last = Input(shape=DIM_a, dtype='float32', name='a_last'); # last step action;
h_a    = GRU(units=DIM_ha)(a_last);
f      = concatenate(axis=-1)(h_m,h_a); # the output cat;
f      = GRU(units=DIM_f)(f);
v_t    = Dense(72, activation='relu')(f);
v_t    = Dense(36, activation='relu')(v_t);
v_t    = Dense(4, activation='relu')(v_t);
v_t    = Dense(1, activation='relu')(v_t);
a_t    = Dense(72, activation='relu')(f);
a_t    = Dense(36, activation='relu')(a_t);
a_t    = Dense(14, activation='relu')(a_t);

model   = Model(inputs=[l,o,a_last], outputs=[v_t,a_t]);
sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True);
model.compile(optimizer=sgd, loss=losses.mean_squared_error, metrics=['accuracy']);



# ========= define your reward func here ==========
def reward():
    pass;

# ========= define your state func here ==========
def state():
	pass;

# ========= define your cost func here ===========
def cost():
    pass;	