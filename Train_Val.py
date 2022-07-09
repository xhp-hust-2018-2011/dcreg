# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import  DataLoader
import torch.nn.functional as F
#from torchvision import models


import os
import numpy as np
from time import time

import math

from utils.IOtools import txt_write 
from Network.class_func import get_local_count


# =============================================================================
# Training Function
# =============================================================================
def train_phase(opt,net,train_loader,optimizer,lossfn_dict,\
    batch_size=1,print_every=10,tepoch=0):
    net.train()
    running_all_loss = 0.0
    Zero_out = torch.zeros(1).squeeze().cuda() if torch.cuda.is_available() \
                else torch.zeros(1).squeeze()
    # auto init loss vlaue and running loss
    running_loss_dict = dict()
    loss_value_dict = dict()
    loss_name_list = list(lossfn_dict.keys())
    for li in range(len(loss_name_list)):
        running_loss_dict[loss_name_list[li]] = 0.0
        loss_value_dict[loss_name_list[li]] = 0.0
    avg_frame_rate = 0.0
    for i, sample in enumerate(train_loader):
        start = time()
        inputs, targets = sample['image'], sample['density_map']
        inputs,targets = inputs.type(torch.float32), targets.type(torch.float32)
        inputs, targets = inputs.cuda(), targets.cuda()
        gt_cmap = get_local_count(targets,opt['W2DOpt']['DenPatchSize'],opt['W2DOpt']['DenPatchSize'])
        targets = net.encode_count(gt_cmap).squeeze(1)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        Cmap = net(im=inputs)
        if lossfn_dict['cls_loss']!=None:
            pre_cls = net.encode_count(Cmap.detach()).squeeze(1)
            mask = (pre_cls.squeeze()!=targets.detach().squeeze()).float()
            loss_value_dict['cls_loss'] = ((gt_cmap-Cmap).squeeze().abs()*mask.squeeze()).sum()/torch.clamp(mask.sum(),min=1.0)
        else:
            loss_value_dict['cls_loss'] = torch.zeros(1).squeeze().cuda()    

        def compute_global_bias_mask(pre_cmap,gt_cmap):
            # pre and gt are all N*1*H*W
            N,_,H,W = gt_cmap.size()
            err_cmap = (pre_cmap.detach()-gt_cmap.detach()).reshape(N,-1)
            im_err = err_cmap.sum(dim=1,keepdim=True)
            # find the index where the major error come from
            err_cmap = torch.sign(im_err)*err_cmap # this ensure all error should be great than 0
            sorted_err, _ = torch.sort(err_cmap,dim=1,descending=False)
            sorted_err_accum = torch.cumsum(sorted_err,dim=1)
            
            thre_index = (sorted_err_accum<=0).long().sum(dim=1)
            thre_index = torch.clamp(thre_index,max = err_cmap.size()[1]-1)
            thre_value = torch.zeros_like(thre_index).float()
            mask = torch.zeros_like(err_cmap).detach()
            for ii in range(N):
                thre_value[ii] = sorted_err[ii,thre_index[ii]]
                if abs(im_err[ii,:])>0.5:
                    mask[ii,:] = (err_cmap[ii,:]>thre_value[ii]).float()
            mask = mask.reshape(N,1,H,W)  
            return mask

        tN,tC = Cmap.size()[0],Cmap.size()[1]
        if lossfn_dict['total_reg_loss']!=None:
            mask = compute_global_bias_mask(Cmap,gt_cmap)
            loss_value_dict['total_reg_loss'] = ((Cmap-gt_cmap).abs()*mask).sum()/torch.clamp(mask.sum(),min=1.0)
        else:
            loss_value_dict['total_reg_loss'] = torch.zeros(1).squeeze().cuda() 

        # auto compute overall loss
        all_loss = 0.0 
        for li in range(len(loss_name_list)):
            all_loss += opt['loss_w'][loss_name_list[li]]\
                *loss_value_dict[loss_name_list[li]]
        # backward + optimize
        all_loss.backward()        
        if not math.isnan(all_loss.item()):
            if not math.isinf(all_loss.item()):
                if opt['cgd']:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 20)
                optimizer.step()
        
        # collect and print statistics
        running_all_loss += all_loss.item()
        # -- auto update loss history
        for li in range(len(loss_name_list)):      
            running_loss_dict[loss_name_list[li]]+=\
                loss_value_dict[loss_name_list[li]].item()

        # torch.cuda.synchronize()
        end = time()

        # auto print
        running_frame_rate = batch_size * float(1 / (end - start))
        avg_frame_rate = (avg_frame_rate*i + running_frame_rate)/(i+1)
        if i %print_every == print_every-1:
            log_str = 'epoch: %d, train: %d/%d, ' % (tepoch, i+1,len(train_loader) )
            for li in range(len(loss_name_list)):
                if lossfn_dict[loss_name_list[li]] == None:
                    continue
                log_str += '%s: %7.5f,' \
                    % (loss_name_list[li],running_loss_dict[loss_name_list[li]]/(i+1))
            log_str+= 'all_loss: %7.5f, frame: %6.2fHz/%6.2fHz\n' \
                %( running_all_loss / (i+1),running_frame_rate,avg_frame_rate)
            print(log_str)
    train_log = log_str
    train_dict=dict()
    train_dict['mae'],train_dict['log'] = running_all_loss / (i+1),train_log
    return train_dict 


def test_phase(opt,net,testloader,log_save_path=None):
    with torch.no_grad():
        net.eval()
        start = time()
        avg_frame_rate = 0
        mae = 0.0;mae_reg = 0.0
        rmse = 0.0;rmse_reg = 0.0
        me = 0.0;me_reg = 0.0
        mae_med = 0.0
        rmse_med = 0.0
        me_med = 0.0


        for j, data in enumerate(testloader):
            inputs,labels = data['image'], data['all_num']
            inputs,labels = inputs.type(torch.float32),labels.type(torch.float32)
            inputs, labels = inputs.cuda(), labels.cuda()
            tmpname = data['name']
            
            density_map = data['density_map']
            psize = opt['W2DOpt']['DenPatchSize']
            count_map = get_local_count(density_map,psize,psize)
            del density_map
            test_cmap =  net.test(test_im=inputs,test_gtden=None)
            test_cmap_med = test_cmap
            pre = test_cmap.sum().item()
            pre_med = test_cmap_med.sum().item()
            gt = labels.sum().item()

            mae += abs(pre-gt);rmse += (pre-gt)*(pre-gt);me += (pre-gt)
            mae_med += abs(pre_med-gt);rmse_med += (pre_med-gt)*(pre_med-gt);me_med += (pre_med-gt)
            end = time()
            running_frame_rate = 1 * float( 1 / (end - start))
            avg_frame_rate = (avg_frame_rate*j + running_frame_rate)/(j+1)
            if j % 1 == 0:    # print every 2000 mini-batches
                print('Test:[%5d/%5d] pre: %.3f gt:%.3f err:%.3f frame: %.2fHz/%.2fHz' %
                        ( j + 1,len(testloader), pre, gt,pre-gt,
                        running_frame_rate,avg_frame_rate) )                    
                start = time()
        
        log_str =  '%10s\t %8s\t &%8s\t &%8s\t\\\\' % (' ','mae','rmse','me')+'\n'
        log_str += '%-10s\t %8.3f\t %8.3f\t %8.3f\t' % ( 'test',mae/(j+1),math.sqrt(rmse/(j+1)),me/(j+1) ) + '\n'
        log_str +=  '%10s\t %8s\t &%8s\t &%8s\t\\\\' % (' ','mae_med','rmse_med','me_med')+'\n'
        log_str += '%-10s\t %8.3f\t %8.3f\t %8.3f\t' % ( 'test',mae_med/(j+1),math.sqrt(rmse_med/(j+1)),me_med/(j+1) ) + '\n'
        print(log_str)
        if log_save_path:
            txt_write(log_save_path,log_str,mode='w')


    im_num = len(testloader)
    test_dict=dict()
    test_dict['mae'] = mae / im_num
    test_dict['mse'] = math.sqrt(rmse/(im_num))
    test_dict['me'] = me/(im_num)    
    test_dict['log'] = log_str

    return test_dict





    






 



