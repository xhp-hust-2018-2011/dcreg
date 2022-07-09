# --------------------------------------------------------
# Search Template Functions
# Written by xhp (xhp2019)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader

import numpy as np
import math
import os

try:
    from class_func import get_local_count
    from load_data_V2 import Countmap_Dataset, ToTensor
except:
    from Network.class_func import get_local_count
    from utils.load_data_V2 import Countmap_Dataset, ToTensor

class get_template_fromdset(object):
    def __init__(self,trainroot,psize=32,save_folder=None):
        self.trainroot = trainroot
        # create train loader
        self.trainset = Countmap_Dataset(self.trainroot,transform=[ToTensor()],\
            LOADMEM=False,TEST = True,DIV=psize,im_rate=1.0)
        self.dlen = len(self.trainset)
        self.psize = psize
        # compute from above setting
        self.save_folder = save_folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
    # cpmpute the maximum count and other count
    def compute_local_count(self):
        patch_count_path = os.path.join(self.save_folder,'patch_count_p%s.pth' %(self.psize))
        if os.path.exists(patch_count_path):
            res = torch.load(patch_count_path)
            # check if psize match
            if res['psize']!=self.psize:
                print('='*60)
                print('Error: Pisze does not match!')
                print('='*60)
            patch_count = res['patch_count']
            patch_max = res['patch_max']       
        else:
            patch_count=[]
            patch_max = []
            for idx in range(self.dlen):
                sample = self.trainset.__getitem__(idx)
                sample['density_map'] = sample['density_map'].float().unsqueeze(1)
                # compute countmap /32 
                sample['density_map'] = get_local_count(sample['density_map'],self.psize,self.psize)
                patch_count.append(sample['density_map'].cpu().view(1,-1))
                patch_max.append(sample['density_map'].cpu().max().item())
                print('%d/%d image patch count has been extracted' %(idx+1,self.dlen))
            # get patch count
            patch_count = torch.cat(patch_count,dim=1)
            patch_count = patch_count.view(-1)
            # get patch max
            patch_max = max(patch_max)
            torch.save({'patch_count':patch_count,'patch_max':patch_max,'psize':self.psize},\
                patch_count_path)


    def ret_patch_gt(self):
        patch_count_path = os.path.join(self.save_folder,'patch_count_p%s.pth' %(self.psize))
        patch_dict = torch.load(patch_count_path)
        patch_count = patch_dict['patch_count']
        return patch_count.view(-1) 

    def ret_patch_max(self):
        Cmax = self.ret_patch_gt().max().item()
        return Cmax



    


if __name__ == "__main__":
    pass