# -*- coding: utf-8 -*-
"""
The final dataset:
1. image   :    64*64;
2. command :     1*32;
3. action  : 1*(3*10);

"""
import os
import os.path
from PIL import Image
import numpy as np
import re

Com = ['STOP','MOVE','TEMP_HIGH','TEMP_NORMAL'];
Obj = ['horse','bed','jeep','sw_pipe','music','cow'];
Act = ['MOVE_FORWARD','MOVE_BACK','MOVE_LEFT','MOVE_RIGHT','STOP','MOVE','TEMP_HIGH','TEMP_NORMAL'];

with open('img_train/Commands.cm', 'r') as f:
    cm_lines = f.read().split('\n');
with open('img_train/Actions.act', 'r') as f:
    act_lines = f.read().split('\n');


VEC_ACT = [];
VEC_CM  = [];
VEC_IMG = [];
for i in range(len(cm_lines)):
#for i in range(10):
    #----- make vec for Command ---------
    Command = cm_lines[i];
    print i
    (com,obj,pos) = re.match(r'go to (.*) the ./obj/box/(.*).obj at (.*);',Command).groups();
    pos  = pos.replace('(','').replace(')','').split(',');
    (pos_x,pos_y) = (pos[0],pos[1]);
    com_vec = np.zeros(len(Com));
    obj_vec = np.zeros(len(Obj));
    com_vec[Com.index(com)] = 1.0;
    obj_vec[Obj.index(obj)] = 1.0;
    VEC_CM.append( com_vec.tolist() + obj_vec.tolist() + [float(pos_x),float(pos_y)] );
    #----- make vec for Action ---------
    Action = act_lines[i][2:-1].split('\',');
    #(act,para) = re.match(r'\'"(.*)"\', "(.*)"',Action[1:-1]).groups();
    (act,para) = (Action[0],Action[1])
    act_vec = np.zeros(len(Act));
    act_vec[Act.index(act)] = 1.0;
    VEC_ACT.append( act_vec.tolist() + [float(para)] );
    #----- make vec for Visual ---------
    img = np.array( Image.open('img_train/'+str(i)+'.png').convert("L").resize((64,64)) );
    VEC_IMG.append(img);
#print VEC_ACT;
#print VEC_CM;

np.save('DATA_CM',np.array(VEC_CM));
np.save('DATA_ACT',np.array(VEC_ACT));
np.save('DATA_IMG',np.array(VEC_IMG));