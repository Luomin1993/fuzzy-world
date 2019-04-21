# -*- coding: utf-8 -*-
import numpy as np;
import re;
from graph import *;
from make_world import *;

path = './gg';

for i in range(12):
    make_world(path,i);
    state = State('Gs');
    state.build_from_fw(path+'/sample_'+str(i)+'.fw');
    state.visualize(path+'/GS_'+str(i)+'.png');
    infer = Inference('Gi');
    infer.build_from_fw(path+'/sample_'+str(i)+'.fw');
    infer.visualize(path+'/GI_'+str(i)+'.png');
