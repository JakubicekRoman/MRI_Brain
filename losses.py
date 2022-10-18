# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 16:52:25 2022

@author: jakubicek
"""

import torch
import numpy as np
import torch.nn.functional as F

def weighted_MSE_valid(output_batch,out_masks):
    
    # _,_,m,n = output_batch.shape
    
    # output_batch = output_batch[:,:,int(0.1*m):int(0.9*m),int(0.1*n):int(0.9*n)]
        
    # mean_frame = torch.mean(output_batch,0,keepdim=True)
    
    # loss = torch.mean( (output_batch -  mean_frame) ** 2)
    
    
    # is_atleast_50 = out_masks > 0
    # is_atleast_50 = torch.mean(is_atleast_50.to(torch.float), 0, keepdim=True) > 0.5
    
    # valid_part = is_atleast_50.repeat(output_batch.size(0),1,1,1) & (out_masks > 0)
    
    
    # out_masks_bin =  (out_masks > 0).to(torch.float)

    
    # out_masks_bin[out_masks_bin == 0] = 0.00001
    
    # out_masks_bin_sum_0 = torch.sum(out_masks_bin,0,keepdim=True)
    
    # weighted_mean_frame = (torch.sum(output_batch*out_masks_bin,0,keepdim=True) / out_masks_bin_sum_0)

    # weighted_mean_frame = weighted_mean_frame.detach()
    
    # weighted_mean_frame = torch.mean(output_batch,0,keepdim=True) 
    
    # tmp = output_batch.clone()
    # tmp[out_masks == 0] = torch.nan
    # weighted_mean_frame = torch.nanmean(tmp,0,keepdim=True)
    
    # loss = torch.sum( (((output_batch -  weighted_mean_frame) ** 2) * out_masks)) / torch.sum(out_masks)
    
    
    
    output_batch = output_batch[:,0,:,:,:].view(output_batch.shape[0],-1)
    out_masks = out_masks[:,0,:,:,:].view(out_masks.shape[0],-1)
    
    is_atleast_50 = out_masks > 0
    is_atleast_50 = torch.mean(is_atleast_50.to(torch.float), 0) > 0.5
    inds = torch.nonzero(is_atleast_50)[:,0]
    
    
    output_batch = output_batch[:,inds]
    out_masks = out_masks[:,inds]
    
    
    out_masks_bin =  (out_masks > 0).to(torch.float)
    weighted_mean_frame = torch.sum(output_batch*out_masks_bin,0,keepdim=True) / torch.sum(out_masks_bin,0,keepdim=True)
    
    loss = torch.mean(torch.sum( (((output_batch -  weighted_mean_frame) ** 2) * out_masks),0) / torch.sum(out_masks,0))
    
    return loss
    


def diagW_regularization(resize_factor, theta, W, device):
    
    eye = torch.eye(theta.shape[1]) * resize_factor
    eye = eye.to(device)
    W = torch.from_numpy(W).to(device)
    
    tmp_matrix_mse = torch.mean((theta - torch.mean(theta,0,keepdim=True))**2,[1,2]).detach()
    # quantile = tmp_matrix_mse >= torch.quantile(tmp_matrix_mse, 0.2)
    loss = torch.mean( torch.mul((torch.mean(theta[:,:,:],0) - eye)**2 , W )  )
    
    return loss


def transl_regularization(tz, device):
    tz = tz.to(device)
    loss = torch.mean( torch.abs(tz) ) 
    return loss



def time_smoothness(output_batch,out_masks,device,neighbours=5):
    
    output_batch_smooth = output_batch.clone()
    
    s = output_batch_smooth.shape
    output_batch_smooth = output_batch_smooth.view(1,1,s[0], s[2],s[3])
    
    mask = torch.ones((1,1,neighbours,1,1)) / neighbours
    mask[:,:,neighbours//2,:,:] = 0
    mask = mask.to(device)
    
    output_batch_smooth = F.conv3d(output_batch_smooth,mask,padding='valid')
    
    ss = output_batch_smooth.shape
    output_batch_smooth = output_batch_smooth.view(ss[2],1,ss[3],ss[4])
    
    out_masks = out_masks[neighbours//2:-(neighbours//2),:,:,:]
    output_batch = output_batch[neighbours//2:-(neighbours//2),:,:,:]

    
    output_batch_smooth = output_batch_smooth[:,0,:,:].view(output_batch_smooth.shape[0],-1)
    output_batch = output_batch[:,0,:,:].view(output_batch.shape[0],-1)
    out_masks = out_masks[:,0,:,:].view(out_masks.shape[0],-1)

    is_atleast_50 = out_masks > 0
    is_atleast_50 = torch.mean(is_atleast_50.to(torch.float), 0) > 0.5
    inds = torch.nonzero(is_atleast_50)[:,0]
    
    
    output_batch = output_batch[:,inds]
    output_batch_smooth = output_batch_smooth[:,inds]
    out_masks = out_masks[:,inds]


    loss = torch.mean(torch.sum( (((output_batch -  output_batch_smooth) ** 2) * out_masks),0) / torch.sum(out_masks,0))
    
    return loss
    
    
                    
                    
                    
                    