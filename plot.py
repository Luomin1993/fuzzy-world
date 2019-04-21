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
<<<<<<< HEAD
    #loss_re = re.compile(r'loss: (.*) -',line);
    #res = re.findall(r'loss: (.*?) -', line);
    #print line
    res = re.findall(r'- loss: (.*?) - task_finish: (.*?)ss', line+'ss');
    if len(res)==0:continue;
    #print res[0];
    loss.append(float(res[0][0]));
    if res[0][1]=='':score.append(0.);continue;
    score.append(float(res[0][1]));
#print score[0:5]    
=======
	#loss_re = re.compile(r'loss: (.*) -',line);
	res = re.findall(r'loss: (.*?) -', line);
	if len(res)==0:continue;
	loss.append(float(res[0]));
	score.append(0.5/loss[-1]);
#print loss;	
>>>>>>> 6a669de68de0c32521c67937141f320f8f02f1c4

def plot(loss,score):
    x  = np.arange(0,len(loss),1)
    y0 = np.array(loss);
    y1 = np.array(score);
<<<<<<< HEAD
    plt.plot(x, y0*10, marker='o',ms=3, mec='r', mfc='w',label=u'Loss $L_{st}$')
    plt.plot(x, y1*100, marker='^', ms=3,label=u'Success rate $S_r$')
=======
    plt.plot(x, y0*10, marker='o',ms=5, mec='r', mfc='w',label=u'Loss of the model $L_m$')
    plt.plot(x, y1, marker='^', ms=4,label=u'Score of the model $S_a$')
>>>>>>> 6a669de68de0c32521c67937141f320f8f02f1c4
    #plt.plot(x, Xrelu(y2,30), marker='+', ms=10,label=u'Image-to-data:Experimental temperature $T_e=300K$')
    plt.legend()  
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
<<<<<<< HEAD
    plt.ylim(-1,80);
    plt.xlim(-2,380);
    plt.xlabel(u"The training batches") 
    plt.ylabel(u"Loss/Score") 
    plt.title("The method: policy gradient") 
=======
    plt.xlabel(u"The training steps") 
    plt.ylabel(u"Loss/Score") 
    plt.title("The model from the paper of Baidu") 
>>>>>>> 6a669de68de0c32521c67937141f320f8f02f1c4
    plt.show() 

plot(loss,score);    