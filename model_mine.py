# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:02:19 2017

@author: Hanss401
"""
import numpy as np;
import re;
from keras.layers import Input, Dense,Conv2D, MaxPooling2D,Flatten;
from keras.layers.merge import concatenate;
from keras.models import Model, load_model;
from keras.models import Sequential;
from keras.layers import Embedding;
from keras.layers import LSTM;
from keras.layers import Bidirectional;
from keras.layers import Dense;
from keras.layers import TimeDistributed;
from keras.layers import Dropout;
from keras.layers import Add;
from keras import backend as K
from keras.layers.recurrent import GRU;
from keras import optimizers;
from keras import losses;
import matplotlib.pyplot as plt
import keras

# ========= define your evaluation here ==========
def mean_pred(y_true, y_pred):
    return K.square(y_pred-y_true);    

# ========= read your dataset here ================
data_l      = np.load('DatGraph/DATA_CM.npy');
data_a      = np.load('DatGraph/DATA_ACT.npy');
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

# ========= define your helper data here ==========
ACTION2WORDS = {};
WORDS_NUM    = 20;
SEN_LEN      = 10;
DIM_IMG      = 64;
WORD_VEC_DIM = 14;
DIM_hm       = 72;
DIM_ha       = 72;
DIM_a        =  9;
DIM_f        = 72;
DIM_COM      = data_l.shape[-1];
ACT_STEPS    = 300;
DIM_Gi       = data_Gi.shape[-1];
DIM_Gs       = data_Gs.shape[-1];
ACT_OUT_DIM  = data_a.shape[-1];

# ========= define model ============
l_in   = Input(shape=(DIM_COM,), dtype='float32', name='l');            # inputed command;
o_in   = Input(shape=(DIM_IMG,DIM_IMG,1), dtype='float32', name='o');   # inputed image;
a_in   = Input(shape=(DIM_a,), dtype='float32', name='a_last');         # last step action;
Gs_in  = Input(shape=(DIM_Gs,), dtype='float32', name='Gs');            # G_s[t];
Gs_in_ = Input(shape=(DIM_Gs,), dtype='float32', name='Gs_');           # G_s[t-1];
Gi_in  = Input(shape=(DIM_Gi,), dtype='float32', name='Gi');            # G_i[t];
#------- Embedding -------
T      = Dense(32, activation='relu')(l_in);
#------- CNN -------------
C      = Conv2D(16, (5,5), strides=(1, 1), activation='relu',padding='same')(o_in); # out:60*60 *6;
C      = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same')(C); # out:30*30 *6;
C      = Conv2D(8, (3,3), strides=(1, 1), activation='relu',padding='same')(C); # out:60*60 *6;
C      = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same')(C);
C      = Conv2D(8, (3,3), strides=(1, 1), activation='relu',padding='same')(C); 
C      = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same')(C);
C      = Flatten()(C); # the output cnn features; 
#------- R(V,l) -------------
#C_     = concatenate([C,T]); # the output cat;
C_1    = Dense(32, activation='relu')(C);
C_2    = Dense(32, activation='relu')(T);
C_     = Add()([C_1,C_2]);
R_S    = Dense(DIM_Gs, activation='relu')(C_); # the output GFT;
#------ I(V,l,G_s[t],G_s[t-1]) ----
C_i    = Dense(DIM_hm, activation='relu')(C_);
S_i    = Dense(DIM_hm, activation='relu')(Gs_in);
S_i_   = Dense(DIM_hm, activation='relu')(Gs_in_);
G_i    = Add()([C_i,S_i,S_i_]);
G_i    = Dense(32, activation='relu')(G_i);
I_G    = Dense(DIM_Gi, activation='relu')(G_i);
#------ GRU for Pi(G_s,G_i,C*) -------
G_S    = Embedding(output_dim=ACT_OUT_DIM, input_dim=DIM_Gs, input_length=ACT_STEPS,name = 'emb1')(Gs_in);
G_I    = Embedding(output_dim=ACT_OUT_DIM, input_dim=DIM_Gi, input_length=ACT_STEPS,name = 'emb2')(Gi_in);
# L      = Embedding(output_dim=ACT_OUT_DIM, input_dim=DIM_COM, input_length=ACT_STEPS,name = 'emb3')(l_in);
G_S    = GRU(units=DIM_ha)(G_S);
G_I    = GRU(units=DIM_ha)(G_I);
G_S    = Dense(ACT_OUT_DIM, activation='relu')(G_S);
G_I    = Dense(ACT_OUT_DIM, activation='relu')(G_I);
h_a    = Add()([G_S,G_I]);
L      = Dense(ACT_OUT_DIM, activation='relu')(l_in);
h_a    = Dense(ACT_OUT_DIM, activation='relu')(h_a);
h_a    = Add()([h_a,L]);
A_out  = Dense(ACT_OUT_DIM, activation='relu')(h_a);



model  = Model(inputs=[Gs_in,Gi_in,l_in,o_in,Gs_in_], outputs=[A_out,R_S,I_G]);
sgd    = optimizers.SGD(lr=0.00001, decay=0.0, momentum=0.9, nesterov=True);
model.compile(optimizer=sgd, loss=losses.mean_squared_error, metrics=['accuracy']);


# ========= train your model here  ================
#print model.layers[17].name;
#for i in range(len(model.layers)):
#    if model.layers[i].name=='emb_1':print i;
#history = LossHistory();
#model.summary();
model.fit([data_Gs,data_Gi,data_l,data_o,data_Gs_], [data_a,data_Gs,data_Gi],epochs=2, batch_size=10);
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')
#绘制acc-loss曲线
#history.loss_plot('epoch')

# ========= define your reward func here ==========
def reward():
    pass;

# ========= define your state func here ==========
def state():
	pass;

# ========= define your cost func here ===========
# Cost = 
def cost_func():
    pass;	

# ========= define regulizer =============
# from keras import regularizers
# model.add(Dense(64, input_dim=64,
#                 kernel_regularizer=regularizers.l2(0.01),
#                 activity_regularizer=regularizers.l1(0.01)))