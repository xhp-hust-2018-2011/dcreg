#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:41:27 2018

@author: xionghaipeng
"""

__author__='xhp'

'''load the dataset'''
#from __future__ import print_function, division
import os
import torch
#import pandas as pd #load csv file
from skimage import io
import numpy as np
#import matplotlib.pyplot as plt
import glob#use glob.glob to get special flielist
import scipy.io as sio#use to import mat as dic,data is ndarray
import cv2 

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset#, DataLoader#
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# =============================================================================
# Dataset Loader
# =============================================================================
class Countmap_Dataset(Dataset):
    """Wheat dataset. also can be used for annotation like density map"""

    def __init__(self,root,im_dir='images',gt_dir='gtdens',\
        transform=None,TEST = False,LOADMEM=False,DIV=32,im_rate=1.0):
        """
        Args:
            img_dir (string ): Directory with all the images.
            tar_dir (string ): Path to the annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.LOADMEM = LOADMEM #whether to load data in memory
        self.LOADFIN = False
        self.image_mem = []
        self.target_mem = []
        self.root = root
        self.im_dir = os.path.join(root,im_dir)
        self.gt_dir = os.path.join(root,gt_dir)
        self.transform = transform
        self.DIV = DIV
        self.rgb = np.array([0.485, 0.456, 0.406]).reshape(1,1,3) #rgbMean is computed from imagenet
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3) 
        # get image list
        self.imlistpath = os.path.join(self.root,'imglist.txt')
        if os.path.exists(self.imlistpath):
            with open(self.imlistpath,'r') as f:
                self.filelist = f.readlines()
                self.filelist = [fpath.strip('\n') for fpath in self.filelist]
            self.filelist = [os.path.join(self.im_dir,fpath) for fpath in self.filelist]
        else:
            im_name = os.path.join(self.im_dir,'*.jpg')
            self.filelist =  glob.glob(im_name)
            with open(self.imlistpath,'w') as f:
                for path in self.filelist:
                    # only save the pure name
                    _,path = os.path.split(path)
                    f.writelines(path+'\n')
        self.ori_dlen = len(self.filelist)
        # sample image list
        self.im_rate = im_rate
        self.dlen = int(self.ori_dlen*im_rate)
        self.filelist = self.filelist[:self.dlen]
        # for test process, load data is different        
        self.TEST = TEST
        if len(self.filelist)<500:
            self.filelist = self.filelist if self.TEST else self.filelist*4#9*self.filelist # this is not good, since the same image sample will appear in a batch
        elif len(self.filelist)<1000:
            self.filelist = self.filelist if self.TEST else self.filelist*2
        else:
            self.filelist = self.filelist if self.TEST else self.filelist
        # self.filelist = self.filelist
        # self.dlen =  len(self.filelist)
        self.dlen =  len(self.filelist)

    def __len__(self):
        return self.dlen

    def __getitem__(self, idx):

        # ------------------------------------
        # 1. see if load from disk or memory
        # ------------------------------------
        if (not self.LOADMEM) or (not self.LOADFIN): 
            img_name =self.filelist[idx]
            image = io.imread(img_name) #load as numpy ndarray
            if len(image.shape)<3:
                image = np.expand_dims(image,axis=2)
            if image.shape[2] == 1 :
                image = np.concatenate((image,image,image),axis=2)
            image = image/255. -self.rgb #to normalization,auto to change dtype
            image = image/self.std # to normalize with std
    
            (filepath,tempfilename) = os.path.split(img_name)
            (name,extension) = os.path.splitext(tempfilename)
            mat_dir = os.path.join( self.gt_dir, '%s.mat' % (name) )
            mat = sio.loadmat(mat_dir)
            # if need to save in memory
            if self.LOADMEM:
                self.image_mem.append(image)
                self.target_mem.append(mat)
                # updata if load finished
                if len(self.image_mem) == self.dlen:
                    self.IF_loadFinished = True
        else:
            image = self.image_mem[idx]
            mat = self.target_mem[idx]
            #target = mat['target']
        
        # collect to sample
        # all_num = mat['all_num'].reshape((1,1))
        all_num = mat['dot_map'].sum().reshape((1,1))
        density_map = mat['density_map']
        sample = {'image': image, 'density_map': density_map,'all_num':all_num,'name':name}
        # To Tensor
        if self.transform:
            for t in self.transform:
                sample = t(sample)
            # sample = self.transform(sample)
        # pad
        sample['image'],sample['density_map'] = \
            get_pad(sample['image'],DIV=self.DIV),get_pad(sample['density_map'],DIV=self.DIV)
        return sample
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        H,W = sample['image'].shape[:2]
        sample['image'] = sample['image'].transpose((2, 0, 1))
        sample['image'] = torch.from_numpy(sample['image'])
        sample['density_map'] = torch.from_numpy(sample['density_map']).view(1,H,W)
        sample['all_num'] = torch.from_numpy(sample['all_num'])
        return sample


def get_pad(inputs,DIV=64):
    h,w = inputs.size()[-2:]
    ph,pw = (DIV-h%DIV),(DIV-w%DIV)
    # print(ph,pw)

    tmp_pad = [0,0,0,0]
    if (ph!=DIV): 
        tmp_pad[2],tmp_pad[3] = ph//2,ph-ph//2
    if (pw!=DIV):
        tmp_pad[0],tmp_pad[1] = pw//2,pw-pw//2
        
    # print(tmp_pad)
    inputs = F.pad(inputs,tmp_pad)

    return inputs

# crop for Tensor
class FixRandCrop9(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        
        H,W = sample['image'].shape[1:3]
        CH,CW = int(H//2),int(W//2)
        CH = 128 if CH<128 else CH
        CW = 128 if CW<128 else CW

        chlist = [0,0,H-CH,H-CH,(H-CH)//2]
        cwlist = [0,W-CW,0,W-CW,(W-CW)//2]

        ch_min = [1,1,(H-CH)//2+1,(H-CH)//2+1]
        ch_max = [(H-CH)//2-1,(H-CH)//2-1,H-CH-1,H-CH-1]
        cw_min = [1,(W-CW)//2+1,1,(W-CW)//2+1]
        cw_max = [(W-CW)//2-1,W-CW-1,(W-CW)//2-1,W-CW-1]
        
        idx = np.random.randint(0,9)
        # crop 5 corner and 4 random patch
        if idx<5:
            th,tw = chlist[idx],cwlist[idx]    
        else:
            idx = idx - 5
            th = np.random.randint(ch_min[idx],ch_max[idx])
            tw = np.random.randint(cw_min[idx],cw_max[idx])


        sample['image'] = sample['image'][:,th:th+CH,tw:tw+CW]
        sample['density_map'] = sample['density_map'][:,th:th+CH,tw:tw+CW]
        return sample

# crop for Tensor
class RandCropFixSize(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,crop_size = 256):
        self.crop_size = crop_size

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W    

        # CH,CW = int(H//2),int(W//2)# half of the image
        # ch,cw = self.crop_size,self.crop_size
        # CH,CW = max(CH,ch),max(CW,cw)
        H,W = sample['image'].shape[1:3]
        CH,CW = self.crop_size,self.crop_size
        if (H>self.crop_size) and (W>self.crop_size):
            # ch_min = [1,1,(H-CH)//2+1,(H-CH)//2+1]
            # ch_max = [(H-CH)//2-1,(H-CH)//2-1,H-CH-1,H-CH-1]
            # cw_min = [1,(W-CW)//2+1,1,(W-CW)//2+1]
            # cw_max = [(W-CW)//2-1,W-CW-1,(W-CW)//2-1,W-CW-1]
            # idx = np.random.randint(0,4)
            # # crop 4 random patch
            # th = np.random.randint(ch_min[idx],ch_max[idx])
            # tw = np.random.randint(cw_min[idx],cw_max[idx])
            th = np.random.randint(0,H-CH+1)# not contain H-CH+1
            tw = np.random.randint(0,W-CW+1)
            th = min(th,H-CH)
            tw = min(tw,W-CW)
        else:
            th,tw = 0,0

        sample['image'] = sample['image'][:,th:th+CH,tw:tw+CW]
        sample['density_map'] = sample['density_map'][:,th:th+CH,tw:tw+CW]
        return sample

class RandFliplr(object):
    def __init__(self,p=0.8):
        self.p = p

    def __call__(self, sample):
        tp = np.random.rand()
        if tp>self.p: 
            sample['image'] = torch.flip(sample['image'],(2,))
            sample['density_map'] = torch.flip(sample['density_map'],(2,))

        return sample

class ResizeMin(object):
    '''
    do resize for numpy, before to Tensor
    '''
    def __init__(self,MinSize=int(256*1.25)):
        self.MinSize = MinSize

    def __call__(self,sample):
        H,W,_ = sample['image'].shape
        minhw = min(H,W)
        if minhw<self.MinSize:
            ratio = float(self.MinSize)/float(minhw)
            nh,nw = int(H*ratio),int(W*ratio)
            # do resize for image
            sample['image'] = cv2.resize(sample['image'],dsize=(nw,nh), \
                interpolation=cv2.INTER_CUBIC)
            # do resize for density map
            ori_num = sample['density_map'].sum().item()
            sample['density_map'] =cv2.resize(sample['density_map'],dsize=(nw,nh),\
                interpolation=cv2.INTER_CUBIC)
            new_num = sample['density_map'].sum().item()
            new_num = max(new_num,1e-6)
            sample['density_map'] = sample['density_map']*ori_num/new_num

        return sample 

class Resize(object):
    '''
    do resize for numpy, before to Tensor
    '''
    def __init__(self,ratio_min=0.6,ratio_max=1.3,p=0.4):
        self.ratio_min = ratio_min
        self.ratio_max =ratio_max
        self.p = p
        
    def __call__(self,sample):
        H,W,_ = sample['image'].shape
        minhw = min(H,W)
        
        tp = np.random.rand()
        # do ratio
        if tp<=self.p:
            ratio = self.ratio_min+(self.ratio_max-self.ratio_min)* np.random.rand()
            nh,nw = int(H*ratio),int(W*ratio)
            # do resize for image
            # sample['image'] = cv2.resize(sample['image'],dsize=(nw,nh), \
            #     interpolation=cv2.INTER_CUBIC)
            sample['image'] = cv2.resize(sample['image'],dsize=(nw,nh), \
                interpolation=cv2.INTER_NEAREST)                
            # do resize for density map
            ori_num = sample['density_map'].sum().item()
            # sample['density_map'] =cv2.resize(sample['density_map'],dsize=(nw,nh),\
            #     interpolation=cv2.INTER_CUBIC)
            sample['density_map'] =cv2.resize(sample['density_map'],dsize=(nw,nh),\
                interpolation=cv2.INTER_NEAREST)
            new_num = sample['density_map'].sum().item()
            new_num = max(new_num,1e-6)
            sample['density_map'] = sample['density_map']*ori_num/new_num

        return sample   


if __name__ =='__main__':
    inputs = torch.ones(6,60,730,970);print('ori_input_size:',str(inputs.size()) )
    inputs = get_pad(inputs);print('pad_input_size:',str(inputs.size()) )
