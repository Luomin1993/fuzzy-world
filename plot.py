# -*-coding:utf-8-*-
import numpy as np;
import os;
import pandas as pd;
import matplotlib.pyplot as plt;
import re;

#=== load data ====
with open('loss.log', 'r') as f:
    loss_lines = f.read().split('\n');

loss  = [];
acc   = [];
score = [];
for line in loss_lines:    
	#loss_re = re.compile(r'loss: (.*) -',line);
	res = re.findall(r'loss: (.*?) -', line);
	if len(res)==0:continue;
	loss.append(float(res[0]));
	score.append(0.5/loss[-1]);
#print loss;	

def plot(loss,score):
    x  = np.arange(0,len(loss),1)
    y0 = np.array(loss);
    y1 = np.array(score);
    plt.plot(x, y0*10, marker='o',ms=5, mec='r', mfc='w',label=u'Loss of the model $L_m$')
    plt.plot(x, y1, marker='^', ms=4,label=u'Score of the model $S_a$')
    #plt.plot(x, Xrelu(y2,30), marker='+', ms=10,label=u'Image-to-data:Experimental temperature $T_e=300K$')
    plt.legend()  
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"The training steps") 
    plt.ylabel(u"Loss/Score") 
    plt.title("The model from the paper of Baidu") 
    plt.show() 

plot(loss,score);    