# Construct infared small object detection dataset
# machao
# 2019-7-3

import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

# 这里还需要根据后面的模型修改这里det的数据结构
# 标签文件格式：
#       teamID      data* frame_count tracking_count
#       frame:19	1	object:1	239	63
def read_label_file(label_file):
    detections = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        items = [txt for txt in lines[0].split()]
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
    return detections

def make_dataset(root, seqs=None):
    if seqs is None:
        seqs = os.listdir(root)
    else:
        seqs = ['data'+str(seq) for seq in seqs]
    images = []
    for seq in seqs:
        seq_dir = os.path.join(root, seq)
        assert os.path.isdir(seq_dir)
        label_file = os.path.join(seq_dir,seq+".txt")
        detections = read_label_file(label_file)
        for root, _, fnames in sorted(os.walk(seq_dir)):
            for fname in fnames:
                if fname.split(".")[1]=='bmp':
                    fileNO = int(fname.split(".")[0])
                    path = os.path.join(root, fname)
                    item = (path, detections[fileNO])
                    images.extend(item)
    return images

def img_loader(img_path, to_3channel=False):
    # opencv的颜色通道顺序为[B,G,R]，而matplotlib的颜色通道顺序为[R,G,B]
    # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    assert img_path.split(".")[-1]=='bmp'
    img = cv2.imread(img_path)
    if not to_3channel:
        img = img[:,:,0]
    return img

class InfaredDataset(Dataset):
    def __init__(self, root, seqs=None, loader=img_loader, to_3channel=False):
        imgs = make_dataset(root, seqs=seqs)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are .bmp"))

        self.root = root
        self.imgs = imgs
        self.loader = img_loader
        self.to_3channel = to_3channel
 
    def __getitem__(self, index):
        path, target = self.imgs[index]
        imgs = self.loader(path, to_3channel=self.to_3channel)
        return imgs, target

    def __len__(self):
        return len(self.imgs)

def get_data_loader(root, batch_size, 
                    seqs=None, 
                    to_3channel=False,
                    shuffle=True, 
                    num_workers=8):

    data_set = InfaredDataset(root, seqs=seqs, to_3channel=to_3channel)
    
    # Data loader for ModelNet view dataset
    data_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    return data_loader
    