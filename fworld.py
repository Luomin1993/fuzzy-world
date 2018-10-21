# -*- coding: utf-8 -*-
import numpy as np;
import re;
import pyglet;
from pyglet.window import key;
import ratcave as rc;

window = pyglet.window.Window()
keys = key.KeyStateHandler()
window.push_handlers(keys)

#================== STATE ======================
def state_stop(world,id):
    if world.objs[id].move_speed==0 and world.objs[id].rot_speed==0 : return True;
    return False;

def state_move(world,id):
    if world.objs[id].move_speed==0 and world.objs[id].rot_speed==0 : return False;
    return True; 

STATE = {"STOP":state_stop,"MOVE":state_move};


#=================== ACTION ======================
def action_move_forward(scene):
    scene.camera.position.z += 2;

def action_move_back(scene):
    scene.camera.position.z -= 2;

ACTION = {"MOVE_FORWARD":action_move_forward,
          "MOVE_BACK":action_move_back}
          #"MOVE_LEFT":action_move_left,
          #"MOVE_RIGHT":action_move_right,
          #"TURN_LEFT":action_turn_left,
          #"STOP":action_stop,
          #"MOVE":move}




class Obj_Attr(object):
    """The Attributes of the object in world"""
    def __init__(self,mesh,position,move_speed,rot_speed,material_id,is_switch=False,turn_on=None):
        super(Obj_Attr, self).__init__()
        self.mesh         =        mesh;
        self.position     =    position;
        self.move_speed   =  move_speed;
        self.rot_speed    =   rot_speed;
        self.material_id  = material_id;
        self.is_switch    =   is_switch;
        self.turn_on      =     turn_on;




name  = 'World';
objs  = [];
objs_num = len(objs);
rules = [];
rules_num = len(rules);
size  = 32;

def make_world_from_fw(fw_path):
    global name,objs,objs_num,rules,rules_num,size;
    f = open(fw_path);
    name     = re.match(r'WORLD_NAME:(.*);',f.readline()).groups()[0];
    size     = int(re.match(r'WORLD_SIZE:(.*);',f.readline()).groups()[0]);
    objs_num = int(re.match(r'OBJ_NUM:(.*);',f.readline()).groups()[0]);
    #objs.append(); # add floor;
    for i in range(objs_num):
        objs.append(resolve_obj( f.readline() ));
    rules_num = int(re.match(r'RULE_NUM:(.*);',f.readline()).groups()[0]);
    for i in range(rules_num):
        rules.append(resolve_rule( f.readline() ));
        
def resolve_rule(line):
    (state_1,state_2) = re.match(r'RULE:(.*)=>(.*);',line).groups();
    state_1 = re.match(r'"(.*)":"(.*)"',state_1).groups();
    state_2 = re.match(r'"(.*)":"(.*)"',state_2).groups();
    return ( (state_1[1],int(state_1[0]) ) ,(state_2[1],int(state_2[0]) ) );

def resolve_obj(line):
    (ID,model,mesh,pos) = re.match(r'OBJ:ID:"(.*)";MODEL:"(.*)";MESH:"(.*)";POS:(.*?);',line).groups();
    print (ID,model,mesh,pos);
    pos = pos.split(',');pos=[float(i) for i in pos];
    return Obj_Attr( rc.WavefrontReader(model).get_mesh(mesh,position=(pos[0], pos[1],-1.5), scale=.03, rotation=(0, 0, 0)),
                     pos,0,0,0);


make_world_from_fw('sample_1.fw');
meshes = [i.mesh for i in objs];
print len(objs);
print 'ok'
scene = rc.Scene(meshes=meshes,camera=rc.Camera(orientation0=(0, 0, -1),rotation=(0, 0, 0)))
scene.bgColor = 33, 33, 33

def move_camera(dt):
    camera_speed = 3
    if keys[key.LEFT]:
        scene.camera.position.x -= camera_speed * dt
    if keys[key.RIGHT]:
        scene.camera.position.x += camera_speed * dt
    if keys[key.UP]:
        scene.camera.position.z += camera_speed * dt
    if keys[key.DOWN]:
        scene.camera.position.z -= camera_speed * dt
    if keys[key.K]:
        scene.camera.position.y += camera_speed * dt
    if keys[key.L]:
        scene.camera.position.y -= camera_speed * dt    
    if keys[key.H]:
        scene.camera.rotation.x += camera_speed * dt
    if keys[key.J]:
        scene.camera.rotation.x -= camera_speed * dt        
pyglet.clock.schedule(move_camera)



@window.event
def on_draw():
    with rc.default_shader:
        scene.draw()

pyglet.app.run();           