import os
import numpy as np
import torch

def interval_divide(patch_count,vmin,vmax,num=50,partition='linear'):
    # this function use all numpy varaiables
    if partition == 'linear':
        step = (vmax-vmin)/(num-1)
        cls_border = np.arange(vmin,vmax,step)
        add = np.array([vmax])
        cls_border = np.concatenate((cls_border,add),axis=0)
    elif partition == 'log':
        step = (np.log(vmax)-np.log(vmin))/(num-1)
        cls_border = np.arange(np.log(vmin),np.log(vmax),step)
        cls_border = np.exp(cls_border)
        add = np.array([vmax])
        cls_border = np.concatenate((cls_border,add),axis=0)  

    

    cls2value = np.zeros(num)
    cls2value[1:] = (cls_border[:-1]+cls_border[1:])/2


    return cls_border,cls2value


