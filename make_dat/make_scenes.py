#!/usr/bin/env python
# coding: utf-8

import numpy as np;
import matplotlib.image as pg;
import cv2;
from PIL import Image;
import re;
import tensorflow as tf;

global_obj_names = ['horse','bed','jeep','pipe','music','cow','table','chair','truck'];
global_relations = ['is bigger than','is smaller than','is the same size as','is on the left of','is on the right of','is upon','is under','is behind','is in front of'];
global_colors    = ['red','green','blue'];
global_commands  = ['move to','move to the left of','move to the right of','move to the front of','move to the behind of','move to the up of'];
move_pos         = [[0,0,0],[0,-1,0],[0,1,0],[1,0,0],[-1,0,0],[0,0,1]];


class OBJ_IMG(object):
    """docstring for OBJ_IMG"""
    def __init__(self, img_arr, lable, bbox, temp, size):
        super(OBJ_IMG, self).__init__()
        self.img_arr = img_arr;
        self.lable   =   lable;
        self.temp    =    temp;
        self.size    =    size;
        self.bbox    =    bbox;    

def command_and_action(pos):
    # obj in air:
    if pos[0]==0:
        x_      = pos[0];
        y_      = pos[1];
        obj_pos = [ y_-2 , 4-x_ , 1];
        c_id    = np.random.randint(5);
        command = global_commands[0:-1][c_id];
        action  = [ obj_pos[0]+move_pos[c_id][0], obj_pos[1]+move_pos[c_id][1], obj_pos[2]];
    # obj not in air:
    if pos[0]>0:
        x_      = pos[0];
        y_      = pos[1];
        obj_pos = [ y_-2 , 4-x_ , 1];
        c_id    = np.random.randint(6);
        command = global_commands[c_id];
        action  = [ obj_pos[0]+move_pos[c_id][0], obj_pos[1]+move_pos[c_id][1], obj_pos[2]+move_pos[c_id][2] ];
    return (command,action);
        
def give_rgba(name):
    arr = pg.imread(name);
    u,v,w = arr.shape;
    for i in range(u):
        for j in range(v):
            if sum(arr[i,j]) == 4.:
                arr[i,j]=np.array([0]*4)
    return arr;            



def paste_alpha(arr,back,bbox):
    for i in range(bbox[0],bbox[0]+arr.shape[0]):
        for j in range(bbox[2],bbox[2]+arr.shape[1]):
            if arr[i-bbox[0]][j-bbox[2]][3]>0:
                if i>399 or j>399:continue;
                back[i][j] = arr[i-bbox[0]][j-bbox[2]][0:4];
    return back;

def find_bbox(arr):
    x_min=arr.shape[0];x_max=0;
    y_min=arr.shape[1];y_max=0;
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j][3]>0:
                if i<y_min:y_min=i;
                if i>y_max:y_max=i;
                if j<x_min:x_min=j;
                if j>x_max:x_max=j;
    return (x_min,x_max,y_min,y_max);            

def pil_bbox(bbox):
    return (bbox[0],bbox[2],bbox[1],bbox[3]);

"""
task list:
1. object detection;
2. detected object classificaction;
3. objects relation detection;
"""
def make_obj_list(list_of_img):
    obj_list = [];
    for name in list_of_img:
        #img = Image.open(name).resize((100,100));
        img = give_rgba(name);
        name_re = re.compile(r'img/(.*?)_(.*?)_(.*?).png');
        (lable,temp,size) = name_re.findall(name)[0];
        bbox = find_bbox(img);
        pg.imsave('tmp.png',img);
        # img_arr = np.array(img.crop(pil_bbox(bbox)));
        img_arr = np.array(Image.open('tmp.png').crop(pil_bbox(bbox)));
        obj_list.append( OBJ_IMG( img_arr, lable, bbox, temp, size)  );
    return obj_list;    

def make_scene(obj_list,scene):
    # size:300*300: netgrid with 3*3;
    unit_size = 100;
    # x_start   = np.random.randint(2);
    # y_start   = np.random.randint(2);
    occupied  = [];
    bboxes    = [];
    sizes     = [];
    obj_names = [];
    for obj in obj_list:
        # obj_names.append(obj.lable);
        pos = (np.random.randint(4),np.random.randint(4));
        while pos in occupied:pos = (np.random.randint(4),np.random.randint(4));
        occupied.append(pos);
        # scene = paste_alpha(obj.img_arr,scene,(unit_size*pos[0]+x_start,0,unit_size*pos[1]+y_start,0  ));
        scene = paste_alpha(obj.img_arr,scene,(unit_size*pos[0],0,unit_size*pos[1],0  ));
        # bboxes.append([unit_size*pos[0]+x_start,unit_size*pos[1]+y_start,obj.bbox[1]-obj.bbox[0],obj.bbox[3]-obj.bbox[2]  ])
        # bboxes.append([unit_size*pos[1]+y_start,unit_size*pos[0]+x_start,obj.bbox[1]-obj.bbox[0],obj.bbox[3]-obj.bbox[2]])
        bboxes.append([unit_size*pos[1],unit_size*pos[0],obj.bbox[1]-obj.bbox[0],obj.bbox[3]-obj.bbox[2]])
        sizes.append([obj.bbox[1]-obj.bbox[0],obj.bbox[3]-obj.bbox[2]]);
    return (scene,occupied,bboxes,sizes);


# def pos_rel(pos_1,pos_2):
#     # up_left(pos_1,pos_2):
#     if pos_1[0]+1==pos_2[0] and pos_1[1]+1==pos_2[1]:return 0;
#     # up_right(pos_1,pos_2):
#     if pos_1[0]-1==pos_2[0] and pos_1[1]+1==pos_2[1]:return 1;
#     # down_left(pos_1,pos_2):
#     if pos_1[0]+1==pos_2[0] and pos_1[1]-1==pos_2[1]:return 2;
#     # down_right(pos_1,pos_2):
#     if pos_1[0]-1==pos_2[0] and pos_1[1]-1==pos_2[1]:return 3;
#     # left(pos_1,pos_2):
#     if pos_1[0]==pos_2[0] and pos_1[1]+1==pos_2[1]:return 4;
#     # right(pos_1,pos_2):
#     if pos_1[0]==pos_2[0] and pos_1[1]-1==pos_2[1]:return 5;
#     # down(pos_1,pos_2):
#     if pos_1[0]-1==pos_2[0] and pos_1[1]==pos_2[1]:return 6;
#     # up(pos_1,pos_2):
#     if pos_1[0]+1==pos_2[0] and pos_1[1]==pos_2[1]:return 7;
#     return 99;


def pos_rel(pos_1,pos_2):
    # behind(pos_1,pos_2):
    if pos_1[0]+1==pos_2[0] and pos_1[1]==pos_2[1] and pos_1[0]>0:return global_relations.index('is behind');
    # before(pos_1,pos_2):
    if pos_1[0]==pos_2[0]+1 and pos_1[1]==pos_2[1] and pos_2[0]>0:return global_relations.index('is in front of');
    # uppon(pos_1,pos_2):
    if pos_1[0]<pos_2[0] and pos_1[0]==0:return global_relations.index('is upon');
    # under(pos_1,pos_2):
    if pos_1[0]>pos_2[0] and pos_2[0]==0:return global_relations.index('is under');
    return 99;
    # ...pass

def size_rel(size_1,size_2):
    # bigger(size_1,size_2):
    if size_1[0]*size_1[1] - size_2[0]*size_2[1]>58:return global_relations.index('is bigger than');
    if size_1[0]*size_1[1] - size_2[0]*size_2[1]<-58:return global_relations.index('is smaller than');
    return global_relations.index('is the same size as');
    # ...pass

def make_scene_png(index):
    objs = ['horse','bed','jeep','pipe','music','cow','table','chair','truck'];
    # f=open('scene/describe.txt','w');
    # with open(file, 'a+') as f:
    relationships = ['up_left','up_right','down_left','down_right','left','right','down','up'];
    colors    = ['red','green','blue'];
    num_objs = np.random.randint(low=3,high=5);
    list_of_img = [];objs_occ=[];objs_colors=[];state = [];objs_pos = [];objs_sizes = [];CA = [];
    describe_txt = '';
    question_txt = '';
    command_txt  = '';
    for i in range(num_objs):
        # f.write( str(line)+'\n' );
        id = np.random.randint(len(objs));
        id_color = np.random.randint(len(colors));
        while id in objs_occ:id = np.random.randint(len(global_obj_names));
        objs_occ.append(id);
        # while not os.path.exists():
        list_of_img.append('img/'+objs[id]+ '_'+ colors[id_color] +'_' +str(np.random.randint(10))+'.png');
        objs_colors.append(id_color);
    scene = np.array(Image.open('./scene'+str(np.random.randint(5))+'.png').resize((400,400)));
    (scene,occupied,bboxes,sizes) = make_scene(make_obj_list(list_of_img),scene)
    for index_1 in range(len(objs_colors)):
        describe_txt += global_obj_names[objs_occ[index_1]] + ' is ' + global_colors[objs_colors[index_1]] + ';';
        question_txt += 'what is the color of ' + global_obj_names[objs_occ[index_1]] + '?';
        objs_pos.append(occupied[index_1]);
        objs_sizes.append(sizes[index_1]);
        # CA.append(command_and_action(occupied[index_1]));
        (command,action) = command_and_action(occupied[index_1]);
        command_txt  += command + ' ' + global_obj_names[objs_occ[index_1]] + ';';
        CA.append(action);
    for index_1 in range(len(occupied)):
        for index_2 in range(len(occupied)):
            if index_2 != index_1:
                p_rel = pos_rel(occupied[index_1],occupied[index_2])
                if p_rel!=99:
                    state.append([p_rel,objs_occ[index_1],objs_occ[index_2]]);
                    describe_txt += global_obj_names[objs_occ[index_1]] + ' ' + global_relations[p_rel] + ' ' + global_obj_names[objs_occ[index_2]] + ';';
                    question_txt += 'what is the position relationship between ' + global_obj_names[objs_occ[index_1]] + ' and ' + global_obj_names[objs_occ[index_2]] + '?';
                s_rel = size_rel(sizes[index_1],sizes[index_2])
                if s_rel!=99:
                    state.append([s_rel,objs_occ[index_1],objs_occ[index_2]]);
                    describe_txt += global_obj_names[objs_occ[index_1]] + ' ' + global_relations[s_rel] + ' ' + global_obj_names[objs_occ[index_2]] + ';';
                    question_txt += 'what is the size relationship between ' + global_obj_names[objs_occ[index_1]] + ' and ' + global_obj_names[objs_occ[index_2]] + '?';
    with open('text/describe.txt', 'a+') as f:
        f.write(describe_txt);
        f.write('\n');
    with open('text/question.txt', 'a+') as f:
        f.write(question_txt);
        f.write('\n');    
    with open('text/command.txt', 'a+') as f:
        f.write(command_txt);
        f.write('\n');        
    Image.fromarray(scene).save('./scene/sc'+str(index)+'.png');
    return (state,bboxes,objs_occ,objs_colors,objs_sizes,objs_pos,CA);            

def make_bbox_mat(BBoxes):
    # Dim = 32*32
    BBoxes_Mat = [];
    for img in BBoxes:
        mat = np.zeros((32,32));
        for bboxes in img:
            x_min = int(float(bboxes[0])*32/300);
            x_max = int(float(bboxes[1])*32/300);
            y_min = int(float(bboxes[2])*32/300);
            y_max = int(float(bboxes[3])*32/300);    
            for i in range(x_min,x_max):
                for j in range(y_min,y_max):
                    mat[i][j] = 1.;
        BBoxes_Mat.append(mat);
    return np.array(BBoxes_Mat);                

def make_scene_many():
    states = [];
    BBoxes = [];
    Objs   = [];
    colors = [];
    Sizes  = [];
    Poses  = [];
    action = [];
    for i in range(30):
        res = make_scene_png(i);
        states.append(res[0]);
        BBoxes.append(res[1]);
        Objs.append(res[2]);
        colors.append(res[3]);
        Sizes.append(res[4]);
        Poses.append(res[5]);
        action.append(res[6]);
    np.save('data_states',np.array(states));    
    np.save('data_bboxes',np.array(BBoxes));
    # np.save('data_bboxes_mat',make_bbox_mat(BBoxes));
    np.save('data_objs',np.array(Objs));
    np.save('data_colors',np.array(colors));
    np.save('data_sizes',np.array(Sizes));
    np.save('data_poses',np.array(Poses));
    np.save('data_act',np.array(action));

# arr = pg.imread('./img/screenshot150.png');
# u,v,w = arr.shape;
# for i in range(u):
#     for j in range(v):
#         if sum(arr[i,j]) == 4.:
#             arr[i,j]=np.array([0]*4)
# pg.imsave('1.png',arr);
#back = pg.imread('./demo.png')#.resize(arr.shape);
#pg.imsave('2.png',arr+back);
#print type(arr)
#car  = Image.fromarray(arr);
#car  = Image.open('./1.png');
#print np.array(car)[:][:][3];
# back = Image.open('./demo.png');
# back = Image.fromarray(paste_alpha(np.array(car),np.array(back),(555,433,555,433)));
# back.show();

# bbox = find_bbox(np.array(car))
# print bbox
# car.crop(pil_bbox(bbox)).show();

# list_of_img = ['img/car_hi_777.png','img/dest_hi_111.png','img/pi_hi_222.png'];
# scene  = np.array(Image.open('./scene0.png').resize((300,300)));
# Image.fromarray(make_scene(make_obj_list(list_of_img),scene)).show();

make_scene_many()