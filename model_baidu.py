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
#======= 写一个LossHistory类，保存loss和acc =======
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

# ========= define your evaluation here ==========
def mean_pred(y_true, y_pred):
    return K.square(y_pred-y_true);    

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
DIM_COM      = 12;
ACT_STEPS    = 300;

# ========= define model ============
l_in   = Input(shape=(DIM_COM,), dtype='float32', name='l');            # inputed command;
o_in   = Input(shape=(DIM_IMG,DIM_IMG,1), dtype='float32', name='o');   # inputed image;
a_in   = Input(shape=(DIM_a,), dtype='float32', name='a_last');         # last step action;
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
#------- GFT -------------
#C_     = concatenate([C,T]); # the output cat;
C_1    = Dense(72, activation='relu')(C);
C_2    = Dense(72, activation='relu')(T);
C_     = Add()([C_1,C_2]);
C_     = Dense(144, activation='relu')(C_);
C_     = Dense(72, activation='relu')(C_); # the output GFT;
#------ GRU of h_m -------
C_     = Embedding(output_dim=72, input_dim=72, input_length=ACT_STEPS,name = 'emb_1')(C_)
h_m    = GRU(units=DIM_hm,return_sequences=True)(C_);
# h_m = LSTM(72, input_dim=72, input_length=ACT_STEPS, return_sequences=True)();
#h_m = Dense(DIM_hm, activation='relu')(C_);
#------ GRU of h_a -------
a_last = Embedding(output_dim=32, input_dim=DIM_a, input_length=ACT_STEPS,name = 'emb_2')(a_in)
h_a    = GRU(units=DIM_ha)(a_last);
#h_a = Dense(DIM_ha, activation='relu')(a_in);
#------ GRU of f ---------
#f      = concatenate([h_m,h_a]); # the output cat;
f_1    = Dense(32, activation='relu')(h_a);
f_2    = Dense(32, activation='relu')(h_m);
f      = Add()([f_1,f_2]);
# f      = Embedding(output_dim=32, input_dim=DIM_ha+DIM_hm, input_length=ACT_STEPS)(f)
f      = GRU(units=DIM_f)(f);
a_t    = Dense(72, activation='relu')(f);
a_t    = Dense(36, activation='relu')(a_t);
a_t    = Dense(DIM_a, activation='relu')(a_t);


# v_t    = Dense(72, activation='relu')(f);
# v_t    = Dense(36, activation='relu')(v_t);
# v_t    = Dense(4, activation='relu')(v_t);
# v_t    = Dense(1, activation='relu')(v_t);
# a_t    = Dense(72, activation='relu')(f);
# a_t    = Dense(36, activation='relu')(a_t);
# a_t    = Dense(14, activation='relu')(a_t);

model  = Model(inputs=[l_in,o_in,a_in], outputs=a_t);
sgd    = optimizers.SGD(lr=0.00001, decay=0.0, momentum=0.9, nesterov=True);
model.layers[17].trainable=False;
model.compile(optimizer=sgd, loss=losses.mean_squared_error, metrics=['accuracy', mean_pred]);

# ========= read your dataset here ================
data_l      =  np.load('DATA_CM.npy');
data_a      = np.load('DATA_ACT.npy');
data_o      = np.load('DATA_IMG.npy').astype('float32');
data_o = np.reshape(data_o, (len(data_o), DIM_IMG,DIM_IMG,1))
#data_a_last = np.array( [np.random.random(DIM_a).tolist()] + data_a[0:-1].tolist()  );  
data_a_last = data_a;data_a_last[1:]=data_a[0:-1];
#print data_a_last[0];

# ========= train your model here  ================
#print model.layers[17].name;
#for i in range(len(model.layers)):
#    if model.layers[i].name=='emb_1':print i;
#history = LossHistory();
model.summary();
#model.fit([data_l,data_o,data_a_last], data_a,epochs=30, batch_size=10);
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

