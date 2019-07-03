# This script is used for codes test, not required strictly!!!
# machao 
# 2019-7-3

import os
import sys
import numpy
import torch


# read label file
def read_label_file(label_file):
    detections = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        items = [txt for txt in lines[0].split()]
        print(items, len(lines))
        assert int(items[2])==len(lines)-1
        for i in range(1, len(lines)): # every line for a frame
            det = []
            line = lines[i].split()
            assert line[0].split(":")[0]=="frame"
            assert int(line[0].split(":")[1])==i-1 # frame count
            assert (int(line[1])*3+2)==len(line) # object count
            for obj_i in range(int(line[1])): # object loop
                obj_txt = line[obj_i*3+2] # object:1
                assert obj_txt.split(":")[0]=="object"
                # assert int(obj_txt.split(":")[1])==obj_i+1 # objectNO
                # obj_ID = int(obj_txt.split(":")[1])
                obj_x = int(line[obj_i*3+3])
                obj_y = int(line[obj_i*3+4])
                det.append((obj_x, obj_y))
            detections.append(det)
    print(detections[-10:])
    return detections

label_file = '/home/machao/Workspace/2019-空天杯/dataset/data1/data1.txt'
read_label_file(label_file)


# show image
import cv2
import matplotlib.pyplot as plt 

def img_loader(img_path, to_3channel=False):
    # opencv的颜色通道顺序为[B,G,R]，而matplotlib的颜色通道顺序为[R,G,B]
    # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(img_path.split(".")[-1])
    assert img_path.split(".")[-1]=='bmp'
    img = cv2.imread(img_path)
    print(img.shape)
    print(sum(sum(img[:,:,0]-img[:,:,1])))
    if not to_3channel:
        img = img[:,:,0]
    print(img.shape)
    return img
    
img_path = '/home/machao/Workspace/2019-空天杯/dataset/data1/1.bmp'
# img = img_loader(img_path, to_3channel=False)
# plt.imshow(img,cmap="gray")
# plt.axis('off')
# plt.show()