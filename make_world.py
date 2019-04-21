# -*- coding: utf-8 -*-
import numpy as np;
import re;
import ratcave as rc;

'''
#----- random num ---------
#INT
import numpy as np
print np.random.randint(low=5, high=10, size=3)

'''

'''make fuzzy-world map based on finite objs and rules'''
<<<<<<< HEAD
def make_world(path,index):
=======
def make_world(index):
>>>>>>> 6a669de68de0c32521c67937141f320f8f02f1c4
    global_objs   = ['horse','bed','jeep','sw_pipe','music','cow','table','chair','truck'];
    global_states = ['TEMP_NORMAL','TEMP_HIGH','MOVE','STOP'];
    #------ pick objs in the world -----------
    objs = [];
    while(True):
        obj = global_objs[np.random.randint(low=0, high=len(global_objs))];
        if obj not in objs:objs.append(obj);
        if len(objs)==5:break;
    #------ make pos in the world ------------
    # x = [-1,-0.5,0,0.5,1];
    # y = [-1,-0.5,0];
    x = [-1.5,-0.75,0,0.75,1.5];
    y = [-1.5,-0.75,0,0.75];
    poses = [];
    poses_in = [];
    for i in range(len(objs)):
        while 1:
            (x_this,y_this) = (np.random.randint(low=0, high=len(x)), np.random.randint(low=0, high=len(y))); 
            if (x_this,y_this) not in poses:poses_in.append((x_this,y_this));poses.append((x[x_this],y[y_this]));break;
            else:continue;
    #------ make rules ---------
    rules = [];
    for i in range(4):
        while 1:
            (id_1,state_1,id_2,state_2) = (np.random.randint(low=0,high=len(objs)),
                                         np.random.randint(low=0,high=len(global_states)),
                                         np.random.randint(low=0,high=len(objs)),
                                         np.random.randint(low=0,high=len(global_states)))
            if ( (id_1,state_1,id_2,state_2) not in rules ) and ( not (id_1==id_2 and state_1==state_2) ):
                rules.append((id_1,state_1,id_2,state_2));break;#print rules[-1];
            else:continue;
    #------ write into file ------
<<<<<<< HEAD
    f = open(path+'/sample_'+str(index)+'.fw', 'w');
=======
    f = open('sample_'+str(index)+'.fw', 'w');
>>>>>>> 6a669de68de0c32521c67937141f320f8f02f1c4
    f.write( 'WORLD_NAME:sam'+str(index)+';\n'   );
    f.write( 'WORLD_SIZE:44;\n' );
    f.write( 'OBJ_NUM:'+str(len(objs))+';\n');
    for i in range(len(objs)):
        f.write( 'OBJ:ID:"'+str(i)+'";MODEL:"./obj/box/'+objs[i]+'.obj";MESH:"'+objs[i]+'";POS:'+str(poses[i][0])+','+str(poses[i][1])+';TEMP:"TEMP_NORMAL";MOVE:"STOP";\n' );
    f.write( 'RULE_NUM:'+str(len(rules))+';\n' )
    for i in range(len(rules)):
        f.write( 'RULE:"'+str(rules[i][0])+'":"'+str(global_states[rules[i][1]])+'"=>"'+str(rules[i][2])+'":"'+str(global_states[rules[i][3]])+'";\n' );    
    f.close();

if __name__ == '__main__':
<<<<<<< HEAD
    make_world(1);        
=======
    make_world(1);        
>>>>>>> 6a669de68de0c32521c67937141f320f8f02f1c4
