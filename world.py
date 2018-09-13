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
def action_donothing(scene):
    pass;

def action_move_forward(scene):
    scene.camera.position.z += .2;

def action_move_back(scene):
    scene.camera.position.z -= .2;

ACTION = {"DO_NOTHING":action_donothing,
          "MOVE_FORWARD":action_move_forward,
          "MOVE_BACK":action_move_back}
          #"MOVE_LEFT":action_move_left,
          #"MOVE_RIGHT":action_move_right,
          #"TURN_LEFT":action_turn_left,
          #"STOP":action_stop,
          #"MOVE":move}

#for key in ACTION:pyglet.clock.schedule(ACTION[key]);


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


class World(object):
    """Wrapper of the scene"""
    def __init__(self):
        super(World, self).__init__();
        self.scene = rc.Scene(meshes=[],camera=rc.Camera(orientation0=(0, 0, -1),rotation=(0, 0, 0)))
        self.scene.bgColor = 33, 33, 33
        self.name  = 'World';
        self.objs  = [];
        self.objs_num = len(self.objs);
        self.rules = [];
        self.rules_num = len(self.rules);
        self.size  = 32;

    def make_world_from_fw(self,fw_path):
        f = open(fw_path);
        self.name     = re.match(r'WORLD_NAME:(.*);',f.readline()).groups()[0];
        self.size     = int(re.match(r'WORLD_SIZE:(.*);',f.readline()).groups()[0]);
        self.objs_num = int(re.match(r'OBJ_NUM:(.*);',f.readline()).groups()[0]);
        self.objs.append(Obj_Attr( rc.WavefrontReader('obj/box/box.obj').get_mesh("box",position=(0, -.1, -1.5), scale=.03*self.size/32, rotation=(0, -90, 0)),
                         (0,0),0,0,0)); # add floor;
        for i in range(self.objs_num):
            self.objs.append(self.resolve_obj( f.readline() ));
        self.rules_num = int(re.match(r'RULE_NUM:(.*);',f.readline()).groups()[0]);
        for i in range(self.rules_num):
            self.rules.append(self.resolve_rule( f.readline() ));
            
    def resolve_rule(self,line):
        (state_1,state_2) = re.match(r'RULE:(.*)=>(.*);',line).groups();
        state_1 = re.match(r'"(.*)":"(.*)"',state_1).groups();
        state_2 = re.match(r'"(.*)":"(.*)"',state_2).groups();
        return ( (state_1[1],int(state_1[0]) ) ,(state_2[1],int(state_2[0]) ) );

    def resolve_obj(self,line):
        (ID,model,mesh,pos) = re.match(r'OBJ:ID:"(.*)";MODEL:"(.*)";MESH:"(.*)";POS:(.*);',line).groups();
        #print (ID,model,mesh,pos);
        pos = pos.split(',');pos=[float(i) for i in pos];
        entity = rc.WavefrontReader(model).get_mesh(mesh,position=(pos[0], .0,pos[1]), scale=.1, rotation=(0, 0, 0));
        entity.uniforms['diffuse'] = 1, 1, 0 #give color;
        return Obj_Attr( entity,pos,0,0,0);

    def add_obj(self):
        pass;  

    @window.event
    def show(self):
        with rc.default_shader:
            self.scene.meshes = [i.mesh for i in self.objs];
            print 'ok'
            self.scene.draw()      



class Agent(object):
    """The agent in the fworld"""
    def __init__(self):
        super(Agent, self).__init__()
        self.actions = None;
        self.reward = 0;
        self.SG     = None; #Semantic Graph;
        self.see    = None;

    def do_action(self):
        return self.actions[np.random.randint(low=0, high=3)];

class Teacher(object):
    """The teacher in the fworld"""
    def __init__(self):
        super(Teacher, self).__init__()
        self.name = None;
        
        

world = World();
world.make_world_from_fw('test.fw');
world.scene.meshes = [i.mesh for i in world.objs];
#print len(world.objs);
#print 'ok'
print world.rules;
SCENE = world.scene;

def move_camera(dt):
    global world;
    camera_speed = 3
    if keys[key.LEFT]:
        world.scene.camera.position.x -= camera_speed * dt
    if keys[key.RIGHT]:
        world.scene.camera.position.x += camera_speed * dt
    if keys[key.UP]:
        world.scene.camera.position.z += camera_speed * dt
    if keys[key.DOWN]:
        world.scene.camera.position.z -= camera_speed * dt
    if keys[key.K]:
        world.scene.camera.position.y += camera_speed * dt
    if keys[key.L]:
        world.scene.camera.position.y -= camera_speed * dt    
    if keys[key.H]:
        world.scene.camera.rotation.x += camera_speed * dt
    if keys[key.J]:
        world.scene.camera.rotation.x -= camera_speed * dt        
pyglet.clock.schedule(move_camera)


def agent_move(dt):
    agent = Agent();
    agent.actions = ['DO_NOTHING','MOVE_FORWARD','MOVE_BACK'];
    ACTION[agent.do_action()](SCENE);
pyglet.clock.schedule(agent_move)

@window.event
def on_draw():
    global world;
    with rc.default_shader:
        SCENE.draw()

pyglet.app.run();           