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

#========= Globle :For One-Hot Vector Making =========
Globle_Objects      = ['jeep', 'music', 'bed', 'cow', 'bed', 'table', 'truck', 'chair', 'tree', 'horse', 'sw_pipe','TEMP_NORMAL','TEMP_HIGH','STOP','MOVE'];
Globle_Predicates   = ['is','LEFT','RIGHT','FRONT','BEHIND','BIGGER','SMALLER'];
Globle_Antagonistic = {'TEMP_NORMAL':'TEMP_HIGH','TEMP_HIGH':'TEMP_NORMAL','STOP':'MOVE','MOVE':'STOP'};
Moves   = ['MOVE_FORWARD','MOVE_BACK','MOVE_LEFT','MOVE_RIGHT'];
Act = ['MOVE_FORWARD','MOVE_BACK','MOVE_LEFT','MOVE_RIGHT','STOP','MOVE','TEMP_HIGH','TEMP_NORMAL'];

with open('DatGraph/Commands.cm', 'r') as f:
    cm_lines = f.read().split('\n')[0:-1];
with open('DatGraph/Actions.act', 'r') as f:
    act_lines = f.read().split('\n')[0:-1];


VEC_ACT = [];
VEC_CM  = [];
VEC_IMG = [];
VEC_QV  = [];
for i in range(len(cm_lines)):
#for i in range(10):
    #----- make vec for Command ---------
    Command = cm_lines[i];
    # print i;
    (com,obj,pos) = re.match(r'go to (.*) the ./obj/box/(.*).obj at (.*);',Command).groups();
    pos = pos.replace('(','').replace(')','').split(',');
    (pos_x,pos_y) = (pos[0],pos[1]);
    # com_vec = np.zeros(len(Com));
    # obj_vec = np.zeros(len(Obj));
    # com_vec[Com.index(com)] = 1.0;
    # obj_vec[Obj.index(obj)] = 1.0;
    vec = np.zeros(len(Globle_Objects));
    vec[Globle_Objects.index(com)] = 1.;
    vec[Globle_Objects.index(obj)] = 1.;
    VEC_CM.append( vec.tolist() + [float(pos_x),float(pos_y)] );
    #----- make vec for Action ---------
    Action = act_lines[i][2:-1].split('\',');
    print Action;
    #(act,para) = re.match(r'\'"(.*)"\', "(.*)"',Action[1:-1]).groups();
    (act,para) = (Action[0],Action[1])
    act_vec = np.zeros(len(Act));
    act_vec[Act.index(act)] = 1.0;
    para_vec = np.zeros(len(Globle_Objects)+1);
    if act in Moves: para_vec[-1] = float(para);
    else:para_vec[Globle_Objects.index(para.replace(' \'','').replace('\'',''))]=1.;
    VEC_ACT.append( act_vec.tolist() + para_vec.tolist() );
    #----- make vec for QValue ---------
    VEC_QV.append( np.array(VEC_ACT[-1])*1.2 );
    #----- make vec for Visual ---------
    img = np.array( Image.open('DatGraph/img/'+str(i)+'.png').convert("L").resize((64,64)) );
    VEC_IMG.append(img);
#print VEC_ACT;
#print VEC_CM;

np.save('DatGraph/DATA_CM',np.array(VEC_CM));
np.save('DatGraph/DATA_ACT',np.array(VEC_ACT));
np.save('DatGraph/DATA_IMG',np.array(VEC_IMG));
np.save('DatGraph/DATA_QV',np.array(VEC_QV));