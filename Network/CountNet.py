import torch.nn as nn
import torch
from torchvision import models

import torch.nn.functional as F
import numpy as np
import math
import os


try:
    from class_func import Count2Class
except:
    from Network.class_func import Count2Class
#import ipdb
# ============================================================================
# 1.base module functions
# ============================================================================ 
# 1.1
def Gauss_initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
# 1.2 
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)   
# ============================================================================
# 2. feature extractor
# ============================================================================ 
# -- 2.1 vgg16 feature extractor
class VGG16_frontend(nn.Module):
    def __init__(self,block_num=5,decode_num=0,Loadweights=True,VGG16bn=True,Freezebn=False):
        super(VGG16_frontend,self).__init__()
        self.block_num = block_num
        self.decode_num = decode_num
        self.Loadweights = Loadweights
        self.VGG16bn = VGG16bn
        self.Freezebn = Freezebn
        block_dict = [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'],\
             [512, 512, 512,'M'], [512, 512, 512,'M']]
        self.frontend_feat = []
        for i in range(block_num):
            self.frontend_feat += block_dict[i]
        if self.VGG16bn:
            self.features = make_layers(self.frontend_feat, batch_norm=True)
        else:
            self.features = make_layers(self.frontend_feat, batch_norm=False)
        if self.Loadweights:
            if self.VGG16bn:
                pretrained_model = models.vgg16_bn(pretrained = True)
            else:
                pretrained_model = models.vgg16(pretrained = True)
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # load the new state dict
            self.load_state_dict(model_dict)
        if self.Freezebn:
            self.freeze_bn()

    def forward(self,x):
        if self.VGG16bn: 
            if self.block_num>=1:
                x = self.features[ 0: 7](x)
            if self.block_num>=2:
                x = self.features[ 7:14](x)
            if self.block_num>=3:    
                x = self.features[ 14:24](x)
            if self.block_num>=4:
                x = self.features[ 24:34](x)
            if self.block_num>=5:
                x = self.features[ 34:44](x)
            conv5_feat =x
        else:
            if self.block_num>=1:
                x = self.features[ 0: 5](x)
            if self.block_num>=2:
                x = self.features[ 5:10](x)
            if self.block_num>=3:    
                x = self.features[ 10:17](x)
            if self.block_num>=4:
                x = self.features[ 17:24](x)
            if self.block_num>=5:
                x = self.features[ 24:31](x)
            conv5_feat =x 
               
        # feature_map = {'conv1':conv1_feat,'conv2': conv2_feat,\
        #     'conv3':conv3_feat,'conv4': conv4_feat, 'conv5': conv5_feat} 
        feature_map = {'conv5': conv5_feat}     
        
        return feature_map

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
   
        
# ============================================================================
# 3. Final CountNet
# ============================================================================ 
class CountNet(nn.Module):
    def __init__(self,EmbedOpt,class_indice,class2count_spec=None):
        super(CountNet,self).__init__()
        self.class_indice = class_indice
        self.cnum = len(self.class_indice)
        self.class2count = np.zeros((self.cnum))
        self.class2count[1:] = (class_indice[:-1]+class_indice[1:])/2
        self.class2count_spec = class2count_spec

        # now we need to generate each class center and interval length
        # they are self.ccen and self.clen, these two should equals to class num
        self.ccen = np.zeros(self.cnum) 
        self.ccen[0] = self.class_indice[0]/2
        self.ccen[1:] =  (self.class_indice[:-1] + self.class_indice[1:])/2
        self.clen = np.zeros(self.cnum)
        self.clen[0] = self.class_indice[0]/2
        self.clen[1:] = (self.class_indice[1:]-self.class_indice[:-1])/2
        # change to tensor
        self.ccen = torch.from_numpy(self.ccen).float()
        self.clen = torch.from_numpy(self.clen).float()

        # Feature Embedding Network
        self.EmbedOpt = EmbedOpt
        self.EmbedNet = VGG16_frontend(block_num=EmbedOpt['block_num'],\
            decode_num=EmbedOpt['decode_num'],Loadweights=EmbedOpt['load_weights'],\
            VGG16bn=EmbedOpt['IF_VGG16bn'],Freezebn=EmbedOpt['IF_freeze_bn'])

        # regression Network
        self.CmapReg = nn.Sequential(
            nn.Conv2d(512,512,(3,3),stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,1,(3,3),stride=1,padding=1),
            nn.ReLU(inplace=True),
        )
        Gauss_initialize_weights(self.CmapReg)


    def forward(self,im):
        '''
        im: 2Bx3xHxw-> 1t:front Bx3xHxW 2r:Behind Bx3xHxW
        cmap: 2Bx1xH/32xW/32
        '''
        # get embed feat
        feat_all = self.EmbedNet(im)
        feat_all = feat_all['conv5']
        # get density map
        Cmap = self.CmapReg(feat_all)
        return Cmap
    

    def test(self,test_im,test_gtden=None):
        # compute feature embedding
        test_feat = self.EmbedNet(test_im)
        test_feat = test_feat['conv5']# 1xCxFHxFW
        # compute the cmap regression result
        test_cmap = self.CmapReg(test_feat)
        test_cmap = test_cmap
        return test_cmap

    # count2class
    def encode_count(self,cmap):
        cmap = Count2Class(cmap,self.class_indice[:-1])
        return cmap


    


