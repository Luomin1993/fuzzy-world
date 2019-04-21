#!/usr/bin/python 
# coding:utf-8 

import matplotlib.pyplot as plt; 
import numpy as np;
from keras.models import Model, load_model;
import sys;
import re;
from PIL import Image,ImageDraw;

# def save_weights():
#     rec_model = load_model('model_reco.h5');
#     gen_model = load_model('model_gene.h5');
#     np.save('V_t',gen_model.get_weights()[-6]);
#     np.save('A_t',rec_model.get_weights()[-6]);
# save_weights();


def batch_test():
    X_img = np.load('visual_data.npy');
    X_img = np.expand_dims(X_img, axis=3);
    Y_A   = np.load('A_semantic_data.npy');
    X_S   = np.load('Q_semantic_data.npy');
    V_t   = np.load('V_t.npy');
    V_t   = np.array(99*[ V_t ]);
    model_dete = load_model('model_reco.h5');
    res = model_dete.predict([X_img[200:299],X_S[200:299],V_t]);
    test = Y_A[200:299];
    right = 0.;
    for i in range(len(res)):
        if res[i].tolist().index(max(res[i].tolist())) == test[i].tolist().index(max(test[i].tolist())):right+=1.;
    print right/len(res);    
batch_test()



# question = sys.argv[1];
# img      = sys.argv[2];

# question = 'What is the positional relationship between chair and horse in the pic';
# img      = './scene/sc58.png';

# objs = ['pipe','chair','truck','horse','music'];
# relationships = ['up_left','up_right','down_left','down_right','left','right','down','up'];
# relationships_ = ['up_left/down_right','up_right/down_left','left/right','down/up'];

# name_re = re.compile(r'What is the positional relationship between (.*?) and (.*?) in the pic');
# (obj1,obj2) = name_re.findall(question)[0];
# print name_re.findall(question);
# model_dete = load_model('model_reco.h5');
# test_img   = np.array(Image.open(img).convert('L').resize((96,96)));
# test_img   = np.expand_dims(test_img, axis=3);
# test_question = np.zeros(5);
# test_question[objs.index(obj1)]=1.;
# test_question[objs.index(obj2)]=1.;
# # test_question   = np.expand_dims(test_question, axis=2);
# res = model_dete.predict([np.array([test_img]),np.array([test_question])])[0];
# # res = [res[0]+res[3],res[1]+res[2],res[4]+res[5],res[6]+res[7]];
# print relationships_[res.tolist().index( max(res) )];

