# IO package
import os
import argparse
from time import time
from torch.nn.modules import loss
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
# Compute package
import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# My IO package
from utils.IOtools import txt_write,get_config_str,load_model,save_model
from utils.load_data_V2 import Countmap_Dataset, ToTensor,RandFliplr,RandCropFixSize,ResizeMin,Resize,FixRandCrop9
# My Network
from Network.CountNet import CountNet
from Network.search_template import get_template_fromdset
# from Network.class_func import get_local_count
# My Train&Val
from Train_Val import train_phase,test_phase
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')

from cls_funcs.Interval_paritition import interval_divide

# fix manual seed for reproducing the result
seed = 42
import random
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
np.random.seed(seed) 
random.seed(seed) 

CUR_DIR = os.path.split(os.path.abspath(__file__))[0]



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Count Classifictaion')   
    parser.add_argument('--dataroot', type=str, default='./data',
                        help='The root path of the dataset')   
    parser.add_argument('--lr', default=1e-5,type=float,
                        help='Original learning rate')
    parser.add_argument('--epoch_num', default=2800,type=int,
                        help='Number of epoch')
    parser.add_argument('--milestones', default=[2500],type=int,nargs='+',
                        help='Number of epoch')
    parser.add_argument('--bs', type=int, default=6,
                        help='batch size in training')
    parser.add_argument('--scale', action='store_true',default=False,
                        help='Scale the dataset')                         
    parser.add_argument('--sdset', type=str, default='JHU',
                        help='Source Dataset')        
    parser.add_argument('--test-only', action='store_true', default = False,
                        help='True for testing only')                                                
    parser.add_argument('--num_workers', type=int, default=2,
                        help='how often to save test result')  
    parser.add_argument('--cgd', action='store_false', default=True,
                        help='crop the gradient')                            
    parser.add_argument('--ptype', type = str,default='log',
                        help='partition_type',choices=['log','linear'])                                                
    return parser.parse_args()

def main(opt):
    DROOT_list = {
        'JHU':'JHU_resize',
        'None':'None'}   
    DATAROOT = opt['dataroot']
    opt['sdname'] = opt['sdset']
    opt['sdroot'] = os.path.join(DATAROOT,DROOT_list[opt['sdname']])
    opt['testdname'] = opt['sdset']
    opt['testdroot'] = os.path.join(DATAROOT,DROOT_list[opt['testdname']])

    # --learning setting
    opt['crop_size'] = 512
    # --Network settinng    
    EmbedOpt,W2DOpt = dict(),dict()
    # ----EmbedOpt
    EmbedOpt['block_num'] = 5
    EmbedOpt['decode_num'] = 0
    EmbedOpt['load_weights'] =True
    EmbedOpt['IF_VGG16bn'] = True
    EmbedOpt['IF_freeze_bn'] = False#True
    # ----W2DOpt
    W2DOpt['DenPatchSize'] = 32
    W2DOpt['EPS'] = 1e-8
    opt['EmbedOpt']=EmbedOpt;opt['W2DOpt']=W2DOpt
    # Total loss function and setting
    # =======================================
    opt['loss_dict'],opt['loss_w'] = dict(),dict()
    loss_name = 'cls_loss' # this time try to limit L1loss comoute in the interval
    opt['loss_dict'][loss_name] = nn.L1Loss(reduction='mean')
    opt['loss_w'][loss_name] = 1.0
    loss_name = 'total_reg_loss' # this is for supervise the total count num
    opt['loss_dict'][loss_name] = nn.L1Loss(reduction='mean')
    opt['loss_w'][loss_name] = 1.00
    # =============================================================================
    # Dataset setting
    # =============================================================================
    # --1.1 dataset setting
    num_workers = opt['num_workers']
    if opt['scale']:
        transform_train = [Resize(ratio_min=0.6,ratio_max=1.3,p=0.25),ResizeMin(MinSize=int(opt['crop_size'])),ToTensor(),RandCropFixSize(crop_size=opt['crop_size']),RandFliplr(p=0.5)]
    else:
        transform_train = [ResizeMin(MinSize=int(opt['crop_size'])),ToTensor(),RandCropFixSize(crop_size=opt['crop_size']),RandFliplr(p=0.5)]

    # -- set save folder 
    save_folder = os.path.join('results',opt['sdname'],'lr_'+str(opt['lr'])\
        +'Ps_'+str(W2DOpt['DenPatchSize']))+'_bs'+str(opt['bs']) 
    save_folder+= '-'+opt['ptype']

    # generate save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)            
    opt['save_folder'] = save_folder


    transform_test = [ToTensor()]
    # train dataset settings
    root = os.path.join(opt['sdroot'],'train')
    trainset = Countmap_Dataset(root,transform=transform_train,\
        LOADMEM=False,TEST = False,DIV=opt['W2DOpt']['DenPatchSize'],im_rate=1.0)
    trainloader = DataLoader(trainset, batch_size=opt['bs'],
                            shuffle=True, num_workers=num_workers)
    # test dataset settings
    root = os.path.join(opt['sdroot'],'test')
    testset = Countmap_Dataset(root,transform=transform_test,\
         LOADMEM=False,TEST = True,DIV=opt['W2DOpt']['DenPatchSize'],im_rate=1.0)
    testloader = DataLoader(testset, batch_size=1,
                            shuffle=False, num_workers=num_workers)
    # =============================================================================
    # Dataset Template setting
    # =============================================================================
    # choose the template of the training set
    root = os.path.join(opt['sdroot'],'train')
    choose_t_func  = get_template_fromdset(root,\
        psize=opt['W2DOpt']['DenPatchSize'],\
        save_folder=CUR_DIR+'/TemplateTongJi-'+opt['sdname'])
    # -- get dataset tongji (max_count and class_indice)
    choose_t_func.compute_local_count()
    # this time use uep partition
    patch_count_gt = choose_t_func.ret_patch_gt()
    Cmax = patch_count_gt.max().item()
    cls_border,cls2value = interval_divide(patch_count_gt,vmin=5e-3,vmax=Cmax,num=100,partition=opt['ptype'])
    opt['class_indice'] = cls_border
    opt['class2count_spec'] = cls2value


    # init networks
    net =CountNet(opt['EmbedOpt'],opt['class_indice'],class2count_spec=opt['class2count_spec']).cuda()   
    optimizer = optim.Adam(net.parameters(),betas=(0.9, 0.999),lr=opt['lr'],weight_decay=5e-4) 
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt['milestones'], gamma=0.1)
   
    # =============================================================================
    # Training & Testing Process
    # =============================================================================
    if opt['test_only']:
        mod_path = 'best_epoch.pth'
        mod_path = os.path.join(CUR_DIR,'models',opt['sdset'],mod_path)
        net,optimizer = load_model(net,optimizer,mod_path)
        log_save_path = 'train-%s_test-%s.txt' %(opt['sdname'],opt['testdname'])
        log_save_path = os.path.join(CUR_DIR,'models',opt['sdset'],log_save_path)
        test_dict = test_phase(opt,net,testloader,log_save_path=log_save_path)
        os._exit(0)

    # --1.2 load the old epoch if exist
    mod_path='tmp_epoch.pth' 
    mod_path=os.path.join(save_folder,mod_path)
    net,optimizer = load_model(net,optimizer,mod_path)
    # =============================================================================
    # print config and save
    # =============================================================================
    config_str=get_config_str(opt)
    print(config_str)
    txt_path = os.path.join(save_folder,'log-all.txt')
    if net.tmp_epoch_num<1:
        txt_write(txt_path,config_str,mode='a') 

    #Train and Val for every epoch (epoch start from 1 rather than 0)
    for epoch in range(net.tmp_epoch_num+1,opt['epoch_num']+1):  # loop over the dataset multiple times 
        scheduler.step()
        net.epoch = epoch
        # =============================================================================
        # train
        # =============================================================================
        train_dict = train_phase(opt,net,trainloader,optimizer,opt['loss_dict'],tepoch=epoch,
            batch_size=opt['bs'],print_every=1)
        net.train_loss.append(train_dict['mae'])
        # =============================================================================
        # test
        # =============================================================================
        test_dict = test_phase(opt,net,testloader,log_save_path=None)
        mae,rmse,me = test_dict['mae'],test_dict['mse'],test_dict['me']
        net.test_loss.append( mae )
        # =============================================================================
        # print log
        # =============================================================================              
        # save results
        mod_path='tmp_epoch.pth' 
        mod_path=os.path.join(save_folder,mod_path)
        save_model(net,optimizer,mod_path)

        def write_log_str():
            # wirte log
            log_str = 'epoch[%6d/%6d]\n' % (epoch,opt['epoch_num'])
            log_str+= '-'*30+'Train log'+'-'*30+'\n'
            log_str+= train_dict['log']
            log_str+= '-'*30+'Test log'+'-'*30+'\n'
            log_str+= test_dict['log']
            min_epoch = net.test_loss.index(min(net.test_loss))+1
            log_str += 'min: %d-epoch mae: %8.3f\n' %(min_epoch,min(net.test_loss))
            log_str+='='*60+'\n'
            print(log_str)
            txt_path = os.path.join(save_folder,'log-all.txt')
            txt_write(txt_path,log_str,mode='a')   
        def plot_lrcurve():        
            # plot learning curves
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(net.train_loss, color='blue',label = 'train loss')
            ax1.set_ylabel('Train Loss')
            ax1.legend(loc = 'upper left')
            ax2 = ax1.twinx()
            ax2.plot(net.test_loss, label = 'test_loss')
            ax2.set_ylabel('Test Loss')
            ax2.legend(loc = 'upper right')
            #plt.plot(net.val_loss, label = 'val loss', color = 'red', linestyle = '--')
            plt.savefig(os.path.join(opt['save_folder'],'learning_curve.jpg'))
            plt.close('all')      
        write_log_str()
        plot_lrcurve()
    print('Finished Training')


if __name__ == '__main__':
    args = parse_args()
    opt = vars(args) 
    main(opt)