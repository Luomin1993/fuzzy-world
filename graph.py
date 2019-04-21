# -*- coding: utf-8 -*-
import numpy as np;
import re;
import matplotlib.pyplot as plt
import networkx as nx


"""
The Recognization Logic Of the Fuzzy World:
---- Graph:
--------  Open(Light)
------------- {LowTemp(room);Write(Amy,Homework)} -> {HighTemp(room);Play(Amy)}
"""

#========= Globle :For One-Hot Vector Making =========
Globle_Objects      = ['jeep', 'music', 'bed', 'cow', 'bed', 'table', 'truck', 'chair', 'tree', 'horse','sw_pipe', 'TEMP_NORMAL','TEMP_HIGH','STOP','MOVE'];
Globle_Predicates   = ['is','LEFT','RIGHT','FRONT','BEHIND','BIGGER','SMALLER'];
Globle_Antagonistic = {'TEMP_NORMAL':'TEMP_HIGH','TEMP_HIGH':'TEMP_NORMAL','STOP':'MOVE','MOVE':'STOP'};

#========= Compare the Size of Two Objs ==============
def read_obj_size(name):
    with open('obj/box/'+name+'.obj') as f:
        lines = f.readlines();
    max_x=0;min_x=0;
    max_y=0;min_y=0;
    max_z=0;min_z=0;
    for line in lines:
        if line[0:2] == 'v ':
            xyz = line[2:].split(' ');
            xyz = [float(i) for i in xyz];
            if xyz[0]>max_x:max_x=xyz[0];
            if xyz[1]>max_y:max_y=xyz[1];
            if xyz[2]>max_z:max_z=xyz[2];
            if xyz[0]<min_x:min_x=xyz[0];
            if xyz[1]<min_y:min_y=xyz[1];
            if xyz[2]<min_z:min_z=xyz[2];
    return abs(max_x-min_x)*abs(max_y-min_y)*abs(max_z-min_z);    


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
        self.predicates = ['LEFT','RIGHT','FRONT','BEHIND','BIGGER','SMALLER'];
        self.attributes = ['TEMP_NORMAL','TEMP_HIGH','STOP','MOVE'];
        self.states_describe = [];
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
        # print 'Objects:'; print self.objects;    
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
        # print 'Positions:'; print self.pos;    
        for i in range(len(self.objects)):
            for j in range(len(self.objects)):
                if i<j:
                    self.obj_pre+=self.pos_relation(i,j,self.pos[i],self.pos[j]);
                    self.obj_pre+=self.size_relation(i,j);
        # print 'objects and predicates:'; print self.obj_pre;                
        con_re = re.compile(r'"(.*)":"(.*)"=>');
        res_re = re.compile(r'=>"(.*)":"(.*)"');
        for line in lines:
            con = con_re.findall(line);
            res = res_re.findall(line);
            if con!=[]:
                con_ = ( int(con[0][0]), self.attributes.index(con[0][1]));
                res_ = ( int(res[0][0]), self.attributes.index(res[0][1]));
                self.events.append(( con_,res_ ));
        # print 'Inference:';print self.events;


    def pos_relation(self,id1,id2,pos1,pos2):
        res = [];
        if pos1[0]<pos2[0]:res.append((id1,0,id2));res.append((id2,1,id1)); # id1 left   id2;
        if pos1[0]>pos2[0]:res.append((id2,0,id1));res.append((id1,1,id2)); # id1 right  id2;
        if pos1[1]<pos2[1]:res.append((id1,3,id2));res.append((id2,2,id1)); # id1 front  id2;
        if pos1[1]>pos2[1]:res.append((id1,2,id2));res.append((id2,3,id1)); # id1 behind id2;
        return res;

    def size_relation(self,id1,id2):
        res = [];
        if read_obj_size(self.objects[id1])<read_obj_size(self.objects[id2]):res.append((id1,5,id2));res.append((id2,4,id1)); # id1 smaller than id2;    
        else:res.append((id1,4,id2));res.append((id2,5,id1)); # id1 bigger  than id2;    
        return res;


    '''
    Objects:
    ['jeep', 'music', 'bed', 'cow', 'bed']
    Positions:
    [(-1.0, -0.5), (0.0, 0.0), (-0.5, 0.0), (-0.5, -1.0), (0.5, -0.5)]
    objects and predicates:
    [(0, 0, 1), (1, 1, 0), (0, 3, 1), (1, 2, 0), (0, 0, 2), (2, 1, 0), (0, 3, 2), (2, 2, 0), (0, 0, 3), (3, 1, 0), (0, 2, 3), (3, 3, 0), (0, 0, 4), (4, 1, 0), (2, 0, 1), (1, 1, 2), (3, 0, 1), (1, 1, 3), (1, 2, 3), (3, 3, 1), (1, 0, 4), (4, 1, 1), (1, 2, 4), (4, 3, 1), (2, 2, 3), (3, 3, 2), (2, 0, 4), (4, 1, 2), (2, 2, 4), (4, 3, 2), (3, 0, 4), (4, 1, 3), (3, 3, 4), (4, 2, 3)]
    Inference:
    [((3, 2), (4, 1)), ((2, 3), (2, 2)), ((0, 0), (3, 1)), ((0, 3), (2, 2))]
    '''
    def make_feature_vec(self):
        d_o = len(Globle_Objects);
        d_p = len(Globle_Predicates);
        self.Mat = np.zeros((d_o,d_o,d_p));
        for pair in self.obj_att:
            pair_str = self.objects[pair[0]] + ' is ' + self.attributes[pair[1]];
            self.states_describe.append(pair_str);
        for pair in self.obj_pre:
            pair_str = self.objects[pair[0]] + ' '+ self.predicates[pair[1]] +' ' + self.objects[pair[2]];
            self.states_describe.append(pair_str);
        # print self.states_describe;                
        # 'bed is TEMP_NORMAL', 'bed is STOP', 'jeep LEFT music', 'music RIGHT jeep', 'jeep BIGGER music',... ...
        for state in self.states_describe:
            state = state.split(' ');
            #self.Mat[Globle_Objects.index(state[0])] = 1.;
            #self.Mat[Globle_Objects.index(state[0])][Globle_Objects.index(state[2])] = 1.;
            self.Mat[Globle_Objects.index(state[0])][Globle_Objects.index(state[2])][Globle_Predicates.index(state[1])] = 1.;
        return self.Mat;

    def make_change(self,obj,att):    
        state = obj + ' is ' + att;
        if state in self.states_describe:return;
        state_anta = obj + ' is ' + Globle_Antagonistic[att];
        del( self.states_describe[ self.states_describe.index(state_anta) ]);
        self.states_describe.append(state);

<<<<<<< HEAD
    def visualize(self,path):
=======
    def visualize(self):
>>>>>>> 6a669de68de0c32521c67937141f320f8f02f1c4
        G = nx.DiGraph();
        attributes = [];
        for edge in self.obj_pre:
            point_0 = self.objects[edge[0]];
            point_1 = self.predicates[edge[1]];
            point_2 = self.objects[edge[2]];
            G.add_edge(point_0, point_1,weight=0.4);
            G.add_edge(point_1, point_2,weight=0.4);
        for pair in self.obj_att:
            G.add_edge(self.objects[pair[0]],self.attributes[pair[1]],weight=0.2);
            attributes.append(self.attributes[pair[1]]);
        pos = nx.spring_layout(G);    
        # nodes
        nx.draw_networkx_nodes(G, pos, nodelist=self.predicates,node_size=1400,node_color='r',node_shape='^');
        nx.draw_networkx_nodes(G, pos, nodelist=self.objects,node_size=1000,node_color='b');
        nx.draw_networkx_nodes(G, pos, nodelist=attributes,node_size=1200,node_color='g',node_shape='>');
        # edges
        nx.draw_networkx_edges(G, pos, style='dashed',arrowsize=13.2,edge_cmap=plt.cm.Blues, width=2,arrows=False);
        # labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif');
        plt.axis('off');
        #plt.show();
<<<<<<< HEAD
        plt.savefig(path);
=======
        plt.savefig("G_s.jpg");
>>>>>>> 6a669de68de0c32521c67937141f320f8f02f1c4
        plt.close('all');

class Inference(object):
    """docstring for Inference"""
    def __init__(self, name):
        super(Inference, self).__init__()
        self.name   = name;
        self.globle_state = State(name);
        self.events = None;
        self.Mat    = None;

    def build_from_fw(self,file_name):
        self.globle_state.build_from_fw(file_name);
        self.events = self.globle_state.events;
        # print self.events;
    
    def make_feature_vec(self):
        self.Mat = np.zeros((len(self.events),len(Globle_Objects)*2));
        for i in range(len(self.events)):
            pair = self.events[i];
            pair_str = [];
            pair_str.append(self.globle_state.objects[pair[0][0]]);
            pair_str.append(self.globle_state.attributes[pair[0][1]]);
            pair_str.append(self.globle_state.objects[pair[1][0]]);
            pair_str.append(self.globle_state.attributes[pair[1][1]]);
            self.Mat[i][Globle_Objects.index(pair_str[0])]                     = 1.;
            self.Mat[i][Globle_Objects.index(pair_str[1])]                     = 1.;
            self.Mat[i][Globle_Objects.index(pair_str[0])+len(Globle_Objects)] = 1.;
            self.Mat[i][Globle_Objects.index(pair_str[1])+len(Globle_Objects)] = 1.;
        #print self.Mat;    
        return self.Mat;

<<<<<<< HEAD
    def visualize(self,path):
=======
    def visualize(self):
>>>>>>> 6a669de68de0c32521c67937141f320f8f02f1c4
        # [((0, 2), (1, 0)), ((4, 3), (2, 3)), ((0, 1), (4, 0)), ((2, 0), (4, 1))];
        G = nx.DiGraph();
        att = [];objs=[];
        for logic in self.events:
            obj_1 = self.globle_state.objects[logic[0][0]];
            obj_2 = self.globle_state.objects[logic[1][0]];
            att_1 = self.globle_state.attributes[logic[0][1]];
            att_2 = self.globle_state.attributes[logic[1][1]];
            if att_1 not in att:att.append(att_1);
            if att_2 not in att:att.append(att_2);            
            if obj_1 not in objs:objs.append(obj_1);
            if obj_2 not in objs:objs.append(obj_2);            
            G.add_edge(obj_1, att_1,weight=0.4);
            G.add_edge(obj_2, att_2,weight=0.4);
            G.add_edge(obj_1, obj_2,weight=0.6);
        pos = nx.spring_layout(G);    
        eatt=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <0.5]
        einf=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.5]
        # nodes
        nx.draw_networkx_nodes(G, pos, nodelist=att,node_size=1400,node_color='r',node_shape='^');
        nx.draw_networkx_nodes(G, pos, nodelist=objs,node_size=1000,node_color='b');
        # edges
        nx.draw_networkx_edges(G, pos, edgelist=eatt,style='dashed',arrowsize=13.2,edge_cmap=plt.cm.Blues, width=2,arrows=False);
        nx.draw_networkx_edges(G, pos, edgelist=einf,arrowsize=13.2,edge_cmap=plt.cm.Blues, width=5);
        # labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif');
        plt.axis('off');
        #plt.show();
<<<<<<< HEAD
        plt.savefig(path);
=======
        plt.savefig("G_I.jpg");
>>>>>>> 6a669de68de0c32521c67937141f320f8f02f1c4
        plt.close('all');

if __name__ == '__main__':
    # state = State('Gs');
    # state.build_from_fw('sample_1.fw');        
    # state.make_feature_vec();
    # state.visualize();
    # print read_obj_size('tree');
    infer = Inference('GI');
    infer.build_from_fw('sample_1.fw');
    infer.make_feature_vec();
    infer.visualize();