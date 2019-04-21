# -*- coding: utf-8 -*-
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



class OBJ_IMG(object):
    """docstring for OBJ_IMG"""
    def __init__(self, img_arr, lable, bbox, temp, size):
        super(OBJ_IMG, self).__init__()
        self.img_arr = img_arr;
        self.lable = lable;
        self.temp  = temp;
        self.size  = size;
        self.bbox  = bbox;    
        
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
                back[i][j] = arr[i-bbox[0]][j-bbox[2]][0:3];
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
    x_start   = np.random.randint(22);
    y_start   = np.random.randint(22);
    occupied  = [];
    bboxes    = [];
    for obj in obj_list:
        pos = (np.random.randint(3),np.random.randint(3));
        while pos in occupied:pos = (np.random.randint(3),np.random.randint(3));
        occupied.append(pos);
        scene = paste_alpha(obj.img_arr,scene,(unit_size*pos[0]+x_start,0,unit_size*pos[1]+y_start,0  ));
        #bboxes.append([unit_size*pos[0]+x_start,unit_size*pos[1]+y_start,obj.bbox[1]-obj.bbox[0],obj.bbox[3]-obj.bbox[2]  ])
        bboxes.append([unit_size*pos[1]+y_start,unit_size*pos[0]+x_start,obj.bbox[1]-obj.bbox[0],obj.bbox[3]-obj.bbox[2]])
    return (scene,occupied,bboxes);


def pos_rel(pos_1,pos_2):
    # up_left(pos_1,pos_2):
    if pos_1[0]+1==pos_2[0] and pos_1[1]+1==pos_2[1]:return 0;
    # up_right(pos_1,pos_2):
    if pos_1[0]-1==pos_2[0] and pos_1[1]+1==pos_2[1]:return 1;
    # down_left(pos_1,pos_2):
    if pos_1[0]+1==pos_2[0] and pos_1[1]-1==pos_2[1]:return 2;
    # down_right(pos_1,pos_2):
    if pos_1[0]-1==pos_2[0] and pos_1[1]-1==pos_2[1]:return 3;
    # left(pos_1,pos_2):
    if pos_1[0]==pos_2[0] and pos_1[1]+1==pos_2[1]:return 4;
    # right(pos_1,pos_2):
    if pos_1[0]==pos_2[0] and pos_1[1]-1==pos_2[1]:return 5;
    # down(pos_1,pos_2):
    if pos_1[0]-1==pos_2[0] and pos_1[1]==pos_2[1]:return 6;
    # up(pos_1,pos_2):
    if pos_1[0]+1==pos_2[0] and pos_1[1]==pos_2[1]:return 7;
    return 99;


def make_scene_png(index):
    objs = ['pipe_normal_','chair_high_','truck_normal_','horse_normal_','music_normal_'];
    # f=open('scene/data.txt','w');
    relationships = ['up_left','up_right','down_left','down_right','left','right','down','up'];
    num_objs = np.random.randint(low=3,high=5);
    list_of_img = [];objs_occ=[];
    state = [];
    for i in range(num_objs):
        # f.write( str(line)+'\n' );
        id = np.random.randint(len(objs));
        while id in objs_occ:id = np.random.randint(len(objs));
        objs_occ.append(id);
        # while not os.path.exists():
        list_of_img.append('img/'+objs[id]+str(np.random.randint(99))+'.png');
    scene = np.array(Image.open('./scene'+str(np.random.randint(7))+'.png').resize((300,300)));
    (scene,occupied,bboxes) = make_scene(make_obj_list(list_of_img),scene)
    for index_1 in range(len(occupied)):
        for index_2 in range(len(occupied)):
            if index_2>index_1:
                rel = pos_rel(occupied[index_1],occupied[index_2])
                if rel!=99:state.append([rel,objs_occ[index_1],objs_occ[index_2]]);
    Image.fromarray(scene).save('./scene/sc'+str(index)+'.png');
    return (state,bboxes,objs_occ);            

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

def make_one_hot(states):
    #semantic = np.zeros((len(states),8+5));
    semantic_one_hot = [];
    for i in range(len(states)):
        all_states_in_graph = []
        for state in states[i]:
            semantic = np.zeros(8+5);
            semantic[state[0]] = 1.;
            semantic[state[1]+8] = 1.;
            semantic[state[2]+8] = 1.;
            all_states_in_graph.append(semantic);
        semantic_one_hot.append(all_states_in_graph);    
    return semantic_one_hot;    

def make_scene_many():
    states = [];
    BBoxes = [];
    Objs   = [];
    for i in range(1000):
        res = make_scene_png(i);
        states.append(res[0]);
        BBoxes.append(res[1]);
        Objs.append(res[2]);
    semantic_one_hot = make_one_hot(states);
    np.save('semantic_one',semantic_one_hot);
    np.save('data_states',np.array(states));    
    np.save('data_bboxes',np.array(BBoxes));
    np.save('data_bboxes_mat',make_bbox_mat(BBoxes));
    np.save('data_objs',np.array(Objs));


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