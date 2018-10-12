# -*- coding: utf-8 -*-

import numpy as np;
import re;

"""
The Recognization Logic Of the Fuzzy World:
---- Graph:
--------  Open(Light)
------------- {LowTemp(room);Write(Amy,Homework)} -> {HighTemp(room);Play(Amy)}
"""


class Predicate(object):
    """
    Predicate : 
    Edge links two States or two Objcts;
    """
    def __init__(self, id):
        super(Predicate, self).__init__()
        self.id = id
        self.name = name;

class Obj(object):
    """
    docstring for Obj
    """
    def __init__(self, id):
        super(Obj, self).__init__()
        self.id = id
        self.name = name;

class Att(object):
    """
    docstring for Att
    """
    def __init__(self, id):
        super(Att, self).__init__()
        self.id = id
        self.name = name;

class State(object):
    """
    which describes the temp state with :
    P = {P_i}; Obj = {Obj_i};
    """
    def __init__(self, id):
        super(State, self).__init__()
        self.id = id;
        self.objects    = [];
        self.predicates = ['LEFT','RIGHT','FRONT','BEHIND'];
        self.attributes = ['TEMP_NORMAL','TEMP_HIGH','STOP','MOVE'];
        self.pos     = [];
        self.obj_pre = [];
        self.obj_att = [];
        self.events  = [];

    def build_from_fw(self,file_name):
        with open(file_name) as f:
            lines = f.readlines();
        # objs_num = int(re.match(r'OBJ_NUM:(.*);',lines[2]).groups()[0]);
        # for line in lines[3:3+objs_num]:
        mesh_re = re.compile(r'MESH:"(.*?)";');
        for line in lines:
            # print lines[3:3+objs_num]
            mesh = mesh_re.findall(line);
            if mesh != []: self.objects.append(mesh[0]);
        # print self.objects;    
        temp_re = re.compile(r'TEMP:"(.*?)";');
        move_re = re.compile(r'MOVE:"(.*?)";');
        for line in lines:
            mesh = mesh_re.findall(line);
            temp = temp_re.findall(line);
            move = move_re.findall(line);
            if mesh != []:
                self.obj_att.append(  (self.objects.index(mesh[0]) , self.attributes.index(temp[0]))  );
                self.obj_att.append(  (self.objects.index(mesh[0]) , self.attributes.index(move[0]))  );
        pos_re = re.compile(r';POS:(.*?),(.*?);');
        for line in lines:
            pos = pos_re.findall(line);
            if pos!=[]:self.pos.append(( float(pos[0][0]),float(pos[0][1]) ));
        # print self.pos;    
        for i in range(len(self.objects)):
            for j in range(len(self.objects)):
                if i<j:
                    self.obj_pre+=self.pos_relation(i,j,self.pos[i],self.pos[j]);
        # print self.obj_pre;                
        con_re = re.compile(r'"(.*)":"(.*)"=>');
        res_re = re.compile(r'=>"(.*)":"(.*)"');
        for line in lines:
            con = con_re.findall(line);
            res = res_re.findall(line);
            if con!=[]:
                con_ = ( int(con[0][0]), self.attributes.index(con[0][1]));
                res_ = ( int(res[0][0]), self.attributes.index(res[0][1]));
                self.events.append(( con_,res_ ));
        print self.events;        


    def pos_relation(self,id1,id2,pos1,pos2):
        res = [];
        if pos1[0]<pos2[0]:res.append((id1,0,id2));res.append((id2,1,id1)); # id1 left   id2;
        if pos1[0]>pos2[0]:res.append((id2,0,id1));res.append((id1,1,id2)); # id1 right  id2;
        if pos1[1]<pos2[1]:res.append((id1,3,id2));res.append((id2,2,id1)); # id1 front  id2;
        if pos1[1]>pos2[1]:res.append((id1,2,id2));res.append((id2,3,id1)); # id1 behind id2;
        return res;



    def make_feature_vec(self):
        pass;            

    def visualize(self):
        pass;    

class Inference(object):
    """docstring for Inference"""
    def __init__(self, name):
        super(Inference, self).__init__()
        self.name   = name;
        self.globle_state = State(name);
        self.events = None;

    def build_from_fw(self,file_name):
        self.globle_state.build_from_fw(file_name);
        self.events = self.globle_state.events;
    
    def make_feature_vec(self):
        pass;

    def visualize(self):
        pass;                            

if __name__ == '__main__':
    state = State('Gs');
    state.build_from_fw('sample_1.fw');        