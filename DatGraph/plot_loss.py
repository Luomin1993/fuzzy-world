# -*-coding:utf-8-*-
import numpy as np;
import os;
import pandas as pd;
import matplotlib.pyplot as plt;
import re;

'''
loss: 136.1294 - dense_14_loss: 0.2341 - dense_4_loss: 38.3315 - dense_9_loss: 97.5638
'''
#=== load data ====
with open('loss.log', 'r') as f:
    loss_lines = f.read().split('\n');

loss0  = [];
loss1  = [];
loss2  = [];
loss3  = [];
loss4  = [];
score  = [];

for line in loss_lines:
    res = re.findall(r'- loss: (.*?) - dense_14_loss: (.*?) - dense_4_loss: (.*?) - dense_9_loss: (.*?) - dense_15_loss: (.*?) - dense_14_task_finish: (.*?) - dense_4', line);
    # res = re.findall(r'loss: (.*?) - dense_14_loss: (.*?) - dense_4_loss: (.*?) - dense_9_loss: (.*?) - dense_14_acc', line);
    if res != []:
        loss0.append(float(res[0][0])); 
        loss1.append(float(res[0][1]));
        loss2.append(float(res[0][2]));
        loss3.append(float(res[0][3]));
        loss4.append(float(res[0][4]));
        score.append(float(res[0][5]));

def plot(loss0):
    x  = np.arange(0,len(loss0),1)
    y0 = np.array(loss0);
    y1 = np.array(loss1);
    y2 = np.array(loss2);
    y3 = np.array(loss3);
    y4 = np.array(loss4);
    y5 = np.array(score);
    plt.plot(x, y0, marker='o',ms=5, mec='r', mfc='w',label=u'Loss $L$')
    plt.plot(x, y1*20, marker='^', ms=4,label=u'Loss of $\pi(a^{(t)}|G_S,G_I,l^{(t)})$')
    plt.plot(x, y2, marker='*', ms=4,label=u'Loss of $R(v^{(t)},l^{(t)})$')
    plt.plot(x, y3, marker='x', ms=4,label=u'Loss of $I(G_S^{(t)},G_S^{(t-1)},v^{(t)})$')
    plt.plot(x, y4*20, marker='^',mec='r',ms=4,label=u'Loss of $Q(A,G_S,G_I,l^{(t)})$')
    plt.plot(x, y5*100, marker='o',mec='b',ms=3,label=u'success rate of $\pi(a^{(t)}|G_S,G_I,l^{(t)})$')
    plt.legend()  
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"The training batches") 
    plt.ylabel(u"Loss") 
    plt.ylim(-1,80);
    plt.xlim(-2,380);
    plt.title("The demo method for the Fuzzy World tasks") 
    plt.show() 

plot(loss0);        