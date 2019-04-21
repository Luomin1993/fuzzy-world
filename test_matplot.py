# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 23:28:29 2017

@author: wyl
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import math
    
# plt.close()  #clf() # 清图  cla() # 清坐标轴 close() # 关窗口
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# ax.axis("equal") #设置图像显示的时候XY轴比例
# plt.grid(True) #添加网格
# plt.ion()  #interactive mode on
# IniObsX=0000
# IniObsY=4000
# IniObsAngle=135
# IniObsSpeed=10*math.sqrt(2)   #米/秒
# print('开始仿真')
# try:
#     for t in range(180):
#         #障碍物船只轨迹
#         #obsX=IniObsX+IniObsSpeed*math.sin(IniObsAngle/180*math.pi)*t
#         obsX=IniObsY+IniObsSpeed*math.cos(IniObsAngle/180*math.pi)*t
#         obsY=100*np.sin(obsX)#IniObsY+IniObsSpeed*math.cos(IniObsAngle/180*math.pi)*t
#         ax.scatter(obsX,obsY,c='b',marker='.')  #散点图
#         #ax.lines.pop(1)  删除轨迹
#         #下面的图,两船的距离
#         plt.pause(0.001)
# except Exception as err:
#     print(err)


# windows = tk.Tk()
# windows.geometry('500x500')  ## 规定窗口大小500*500像素
# windows.resizable(False, False)  ## 规定窗口不可缩放
# lab1 = tk.Label(windows, text='lab1', height=1, width=15, bg='blue', fg='white')
# lab1.grid(row=0, column=0, padx=5, pady=5)
# lab2 = tk.Label(windows, text='lab2', height=1, width=15, bg='blue', fg='white')
# lab2.grid(row=0, column=1, ipadx=5, ipady=5)
# lab3 = tk.Label(windows, text='lab3', height=1, width=15, bg='red', fg='white')
# lab3.grid(row=0, column=2)
# windows.mainloop()    

from Tkinter import *
  
def onGo():        
    def counter(i):        
        if i > 0:
            loss_train=np.sin(i);accuracy_train=np.cos(i);
            textvar = "Step:%3d,Train_loss:%9g,Train_accuracy: %g" %(i, loss_train, accuracy_train)
            textbox.insert(END, str(textvar+'\n') )
            textbox.after(1000, counter, i-1)
        else:
            goBtn.config(state=NORMAL)
    goBtn.config(state=DISABLED)
    counter(50)

i=50;                  
root = Tk()
textbox = Text(root)
textbox.pack()
goBtn = Button(text = "Go!",command = onGo)
goBtn.pack()
root.mainloop()


# root = Tk()
# texbox = Text(root, height=2, width=30)
# texbox.pack()
# texbox.insert(END, "Just a text Widget\nin two lines\n")
# mainloop()

# for i in range(100):
#     #loss_train=np.sin(i);accuracy_train=np.cos(i);
#     #textvar = "Step:%3d,Train_loss:%9g,Train_accuracy: %g" %(i, loss_train, accuracy_train)
#     #texbox.insert(END, str(textvar+'\n'))
#     #texbox.insert(END, '111')
#     texbox.update()