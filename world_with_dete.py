# -*- coding: utf-8 -*-
import numpy as np;
import re;
import pyglet;
from pyglet.window import key;
import ratcave as rc;
from PIL import Image;
import cv2;
from scipy.ndimage import filters;
import matplotlib.pyplot as plt
from matplotlib.patches import Circle;
import detection as dete;

window = pyglet.window.Window(width=300, height=300,caption='Fullscreen');
window.set_location(555, 333);
#window.set_fullscreen(fullscreen=True, width=800, height=800);
keys = key.KeyStateHandler();
window.push_handlers(keys);

#============= take screenshot ==============
times = 0;
@window.event
def take_screenshot(dt):
    """ takes a screenshot of the client size and saves the image """
    global times;times+=1;
    if times%10==0:pyglet.image.get_buffer_manager().get_color_buffer().save('img/screenshot'+str(times)+'.png');
pyglet.clock.schedule(take_screenshot);


#============ give filter show ====================
# def take_filter(dt):
#     """ give filter show of saved images """
#     global times;
#     if times == 0 or times ==10:return;
#     if times%10==0: 
#         im   = np.array(Image.open('img/screenshot'+str(times-10)+'.png').convert('L'));
#         imx  = np.zeros(im.shape);filters.sobel(im,1,imx);
#         imy  = np.zeros(im.shape);filters.sobel(im,0,imy);
#         imxy = np.sqrt(imx**2+imy**2);
#         Image.fromarray( np.append(np.append(imx,imy,axis=1),imxy,axis=1)  ).convert('RGB').save('img_filter/f'+str(times)+'.jpg');
#         cv2.imshow('time',cv.imread('img_filter/f'+str(times)+'.jpg'));cv.waitKey(52);
# pyglet.clock.schedule(take_filter);

#============ give filter show ====================
def give_dete(dt):
    """ give dete show of saved images """
    global times;
    if times == 0 or times ==10:return;
    if times%10==0: 
        img   = np.array(Image.open('img/screenshot'+str(times-10)+'.png').convert('L'));
        #---- draw bboxes ---------
        bboxes = dete.give_dete_bbox();
        for bbox in bboxes:
            cv2.rectangle(img,bbox[0], bbox[1],(255,0,0),2);
        #---- draw texts ---------
        texts = dete.give_class();
        for i in range(len(texts)):
            cv2.putText(img,texts[i],bboxes[i][0],cv2.FONT_HERSHEY_PLAIN,1.4,(111,111,255),1);
        #---- draw lines ---------
        states = dete.give_states();
        # print bboxes
        for i in range(len(states)):
            # w1     = bboxes[obj1][2];
            # h1     = bboxes[states[i][1]][3];
            # w2     = bboxes[obj2][2];
            # h2     = bboxes[states[i][2]][3];
            obj1 = states[i][1];
            obj2 = states[i][2];
            point1 = ( (bboxes[obj1][0][0]+bboxes[obj1][1][0])/2 ,(bboxes[obj1][0][1]+bboxes[obj1][1][1])/2);
            point2 = ( (bboxes[obj2][0][0]+bboxes[obj2][1][0])/2 ,(bboxes[obj2][0][1]+bboxes[obj2][1][1])/2);
            cv2.line(img, point1, point2, (111, 255,111),1);
            cv2.putText(img,states[i][0],(  (point1[0]+point2[0])/2,  (point1[1]+point2[1])/2),cv2.FONT_HERSHEY_PLAIN,1.4,(222,222,11),1);
        #---- save and show -------
        Image.fromarray( img  ).convert('RGB').save('img_filter/f'+str(times)+'.jpg');
        cv2.imshow('time',cv2.imread('img_filter/f'+str(times)+'.jpg'));cv2.waitKey(52);
pyglet.clock.schedule(give_dete);


#============== write your helper funcs here ==============
def give_color_by_temp(temp):
    if temp=='TEMP_HIGH':return 0,0,1;
    if temp=='TEMP_NORMAL':return 0,1,0;
    return 1,0,0;


#=============== judge STATE ==============
def is_state_stop(obj):
    return obj.move == 'STOP';

def is_state_move(obj):
    return obj.move == 'MOVE';

def is_state_turnon(obj):
    return obj.is_switch==1 and obj.turnon == 'TURN_ON';

def is_state_turnoff(obj):
    return obj.is_switch==1 and obj.turnon == 'TURN_OFF';

def is_state_temphigh(obj):
    return obj.temp == 'TEMP_HIGH';    

def is_state_tempnormal(obj):
    return obj.temp == 'TEMP_NORMAL';

STATE_JUDGE = {"STOP":is_state_stop,"MOVE":is_state_move,"TURN_ON":is_state_turnon,"TURN_OFF":is_state_turnoff,"TEMP_HIGH":is_state_temphigh,"TEMP_NORMAL":is_state_tempnormal};

#================== set STATE ==============
def set_state_stop(obj):
    obj.move = 'STOP';return obj;

def set_state_move(obj):
    obj.move = 'MOVE';return obj;

def set_state_turnon(obj):
    if not obj.is_switch:return obj;
    obj.turnon = 'TURN_ON';return obj;

def set_state_turnoff(obj):
    if not obj.is_switch:return obj;
    obj.turnon = 'TURN_OFF';return obj;

def set_state_temphigh(obj):
    obj.temp = 'TEMP_HIGH';return obj;

def set_state_tempnormal(obj):
    obj.temp = 'TEMP_NORMAL';return obj;    

STATE_SET = {"STOP":set_state_stop,"MOVE":set_state_move,"TURN_ON":set_state_turnon,"TURN_OFF":set_state_turnoff,"TEMP_HIGH":set_state_temphigh,"TEMP_NORMAL":set_state_tempnormal};

#================== STATE implementation ======================
def state_stop(obj,dt):
    #if world.objs[id].move_speed==0 and world.objs[id].rot_speed==0 : return True;
    #return False;
    obj.mesh.rotation.y += 0*dt;

def state_move(obj,dt):
    #if world.objs[id].move_speed==0 and world.objs[id].rot_speed==0 : return False;
    #return True; 
    obj.mesh.rotation.y += 12*dt;

def state_temphigh(obj,dt):
    obj.mesh.uniforms['diffuse'] = give_color_by_temp(obj.temp);

STATE_IM = {"STOP":state_stop,"MOVE":state_move,"TEMP_HIGH":state_temphigh,"TEMP_NORMAL":state_temphigh};



#=================== ACTION ======================
def action_donothing(scene):
    pass;

def action_move_forward(scene):
    scene.camera.position.z += .2;

def action_move_back(scene):
    scene.camera.position.z -= .2;
 

ACTION = {"DO_NOTHING":action_donothing,
          "MOVE_FORWARD":action_move_forward,
          "MOVE_BACK":action_move_back,
          "STOP":set_state_stop,
          "MOVE":set_state_move,
          "TEMP_HIGH":set_state_temphigh,
          "TEMP_NORMAL":set_state_tempnormal};
          #"MOVE_LEFT":action_move_left,
          #"MOVE_RIGHT":action_move_right,
          #"TURN_LEFT":action_turn_left,
          #"MOVE":move}

class Obj_Attr(object):
    """The Attributes of the object in world"""
    def __init__(self,mesh,position,move,temp,material_id,is_switch=False,turn_on='TURN_OFF'):
        super(Obj_Attr, self).__init__()
        self.mesh         =          mesh;
        self.position     =      position;
        self.move         =          move;
        self.temp         =          temp;  
        self.material_id  =   material_id;
        self.is_switch    =     is_switch;
        self.turnon       =       turn_on;


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
        self.objs.append(Obj_Attr( rc.WavefrontReader('obj/box/box.obj').get_mesh("box",position=(0, -.1, -1.5), scale=.03*self.size/20, rotation=(0, -90, 0)),
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
        (ID,model,mesh,pos,temp,move) = re.match(r'OBJ:ID:"(.*)";MODEL:"(.*)";MESH:"(.*)";POS:(.*);TEMP:"(.*)";MOVE:"(.*)";',line).groups();
        #print (ID,model,mesh,pos);
        pos = pos.split(',');pos=[float(i) for i in pos];
        entity = rc.WavefrontReader(model).get_mesh(mesh,position=(pos[0], -.1,pos[1]), scale=.1, rotation=(0, 0, 0));
        #entity.uniforms['diffuse'] = 1, 1, 0 #give color;
        return Obj_Attr(entity,pos,move,temp,0);

    def make_world_to_fw(self,fw_path):
        pass;
    
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
        self.actions  = None;
        self.reward   = 0;
        self.SG       = None; #Semantic Graph;
        self.see      = None;

    def do_action(self,ACT_ID,OBJ_ID):
        #return self.actions[np.random.randint(low=0, high=3)];
        global world;
        ACTION[ACT_ID](world.objs[OBJ_ID]);

class Teacher(object):
    """The teacher in the fworld"""
    def __init__(self):
        super(Teacher, self).__init__()
        self.name = None;
        
        

world = World();
world.make_world_from_fw('gg/sample_11.fw');
world.scene.camera.position.y += 0.2; 
world.scene.camera.rotation.x -= 40;
world.scene.meshes = [i.mesh for i in world.objs];
#print len(world.objs);
#print 'ok'
#print world.rules;
#SCENE = world.scene;

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
        world.scene.camera.rotation.x += 13 * dt
    if keys[key.J]:
        world.scene.camera.rotation.x -= 13 * dt        
    if keys[key.A]:
        world.scene.camera.rotation.y += 13 * dt;#world.scene.camera.rotation.z += 13 * dt;
    if keys[key.D]:
        world.scene.camera.rotation.y -= 13 * dt;#world.scene.camera.rotation.z -= 13 * dt;    
pyglet.clock.schedule(move_camera);

#----- draw the objects animations and states in the scene ------------
def __draw__(dt):
    # =========== Follow Rule ===========
    for rule in world.rules:
        #[(('STOP', 1), ('MOVE', 2))]
        if STATE_JUDGE[rule[0][0]]( world.objs[ int(rule[0][1]) ] ): 
            world.objs[ int(rule[1][1]) ] = STATE_SET[rule[1][0]](world.objs[ int(rule[1][1]) ]);
    # =========== Follow State ==========
    for obj in world.objs[1:]:
        STATE_IM[obj.move](obj,dt);
        STATE_IM[obj.temp](obj,dt);        
pyglet.clock.schedule(__draw__);


#----- agent navigation ---------
# def agent_move(dt):
#     agent = Agent();
#     agent.actions = ['DO_NOTHING','MOVE_FORWARD','MOVE_BACK'];
#     ACTION[agent.do_action()](world.scene);
# pyglet.clock.schedule(agent_move)

#========== plot dynamicly ==========
def draw_dynamic():
    pass; 


#----- agent take action and trigger rule-------
#----- Write your RL algorithm here ------------
time_passed = 0;
agent = Agent();
#--- plot context ---
# fig=plt.figure();
# ax=fig.add_subplot(1,1,1);
# ax.axis("equal");
# plt.grid(True);
# plt.ion();
#--- report context ---      
# print('\033[1;33;44m If the torus is stopped then the monkey can move \033[0m')
def agent_act(dt):
    global time_passed,agent;
    loss_1=np.sin(time_passed);loss_2=np.cos(time_passed);loss_3=np.tan(time_passed);
    #print(' \033[0;31m STEP: \033[0m  '+str(time_passed)+' \033[0;31m Loss: \033[0m'+str(loss_1)+' \033[0;31m Loss: \033[0m'+str(loss_2)+' \033[0;31m Loss: \033[0m'+str(loss_3)) 
    #ax.scatter(time_passed,loss_1,c='b',marker='.')  #散点图
    #plt.pause(0.001)
    time_passed+=dt;#print time_passed;
    #if time_passed>4:agent.do_action("MOVE",2);agent.do_action("TEMP_HIGH",2);
pyglet.clock.schedule(agent_act)

@window.event
def on_draw():
    global world;
    with rc.default_shader:
        world.scene.draw()

pyglet.app.run();