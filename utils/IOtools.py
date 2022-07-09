# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 20:06:33 2018

@author: xhp
"""
# =============================================================================
# 1. Txt tools
# =============================================================================
import os 
import copy
import torch

# Func 1.1: write txt file ( append  or rewrite )
def txt_write(file_name,str,mode='a'):
    if not os.path.exists(file_name):
        mode = 'w'

    with open(file_name,mode) as f:
        f.write(str)

# get opt string as config
def get_config_str(opt):
    outstr = '-'*60 +'\n'
    outstr += '---Configuration---\n'
    # get print each key of the opt (dict)
    for key in list(opt.keys()):
        tmpstr = '%s:\t %s\n' %(key,str(opt[key]))
        outstr+=tmpstr
    outstr += '-'*60 +'\n'
    return outstr


def load_model(net,optimizer,mod_path,Pretrained=False):
    if os.path.exists(mod_path):
        all_state_dict = torch.load(mod_path)
        pretrained_dict = all_state_dict['net_state_dict']
        model_dict = net.state_dict()
        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # load the new state dict
        net.load_state_dict(model_dict)

        net.epoch = 0
        if not Pretrained:
            net.train_loss = all_state_dict['train_loss']
            net.val_loss = all_state_dict['val_loss']
            net.test_loss = all_state_dict['test_loss']
            net.tmp_epoch_num = all_state_dict['tmp_epoch_num']
            net.epoch = net.tmp_epoch_num
            print('Load: %d-th-epoch' %(net.tmp_epoch_num))
        else:
            net.train_loss = []; net.val_loss = []; net.test_loss = []
            net.tmp_epoch_num = 0
            print('Load: %d-th-epoch as pretrained' %(all_state_dict['tmp_epoch_num']))
            print('Begin training from 1st epoch')
        del all_state_dict
    else:
        net.tmp_epoch_num = 0
        net.train_loss = []; net.val_loss = []; net.test_loss = []
        print('No Model loaded!')
    return net,optimizer

def save_model(net,optimizer,mod_path):
    torch.save({'tmp_epoch_num':net.epoch,
    'net_state_dict':net.state_dict(),
    'train_loss':net.train_loss,
    'val_loss':net.val_loss,
    'test_loss':net.test_loss},
    mod_path)

    # save the best model
    if (net.test_loss[-1]==min(net.test_loss)):
        torch.save({'tmp_epoch_num':net.epoch,
        'net_state_dict':net.state_dict(),
        'train_loss':net.train_loss,
        'val_loss':net.val_loss,
        'test_loss':net.test_loss},
        mod_path.replace('tmp_epoch.pth','best_epoch.pth'))
    
