# -*- coding: utf-8 -*-
"""
Created on Fri Nov 5 16:02:19 2018

@author: Hanss401
"""
import numpy as np;
import re;

def give_dete_bbox():
    bboxes  = []
    (x1,y1) = ( np.random.randint(low=10,high=20),np.random.randint(low=10,high=20) )
    (w1,h1) = (np.random.randint(low=75,high=90),np.random.randint(low=75,high=90))
    bboxes.append([(x1,y1),(x1+w1,y1+h1)]);
    (x2,y2) = (np.random.randint(low=40,high=50),np.random.randint(low=250,high=270))
    (w2,h2) = (np.random.randint(low=75,high=90),np.random.randint(low=75,high=90))
    bboxes.append([(x2,y2),(x2+w2,y2+h2)]);
    (x3,y3) = (np.random.randint(low=240,high=250),np.random.randint(low=80,high=100))
    (w3,h3) = (np.random.randint(low=75,high=90),np.random.randint(low=75,high=90))
    bboxes.append([(x3,y3),(x3+w3,y3+h3)]);
    (x4,y4) = (np.random.randint(low=240,high=250),np.random.randint(low=220,high=250))
    (w4,h4) = (np.random.randint(low=75,high=90),np.random.randint(low=75,high=90))
    bboxes.append([(x4,y4),(x4+w4,y4+h4)]);
    return bboxes;


def give_class():
    res  = [];
    objs = ['pipe','chair','truck','horse','music'];    
    for i in range(4):
        res.append( objs[np.random.randint(4)] );
    return res;   

def give_states():
    states = [];
    relationships = ['up_left','up_right','down_left','down_right','left','right','down','up']; 
    occ = []
    while len(states)<=4:
    	obj1 = np.random.randint(4);
    	obj2 = np.random.randint(4);
    	while obj2==obj1:obj2=np.random.randint(4);
    	if ((obj1,obj2) in occ) or ((obj2,obj1) in occ):continue;
    	occ.append((obj1,obj2));occ.append((obj2,obj1));
    	states.append([relationships[ np.random.randint(7) ],obj1,obj2]);
    return states;	