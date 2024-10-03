# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:39:55 2021

@author: Zc
"""

import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class BDDC_2D(nn.Module):
 

    def __init__(self, miu):
        super().__init__()
        self.miu = nn.Parameter(torch.tensor([miu]), requires_grad=True)
   
    # Forward is a CG solving the objective function. 
    
    # notice the zero-filled and z are 2-channel-splited real 
    def forward(self, z, kspace, kernel, spirit_kernels,prev_kspace):
        #prev_kspace = kspace    
        divp = kspace.shape[3]//2
        kk = kspace[0,:,:,0:divp] + kspace[0,:,:,divp:]*1j
        m, n, no_c = kk.shape
        locs = torch.abs(kk) > 0

        DTy = torch.unsqueeze(torch.reshape(self.adjoint_selection_operator(self.selection_operator(torch.reshape(kk,(m * n * no_c,1)).cuda(), locs, m, n, no_c), locs, m, n, no_c),(m,n,no_c)),dim=0)

        rho = 1
        def ATA(x,kernel,spirit_kernels,locs,m,n,no_c):
            a1 = self.selection_operator(x[0,...], locs, m, n, no_c).cuda()
            a2 = self.adjoint_selection_operator(a1, locs, m, n, no_c).cuda()
                
            b1 = self.conv_forward_new(x[0,...], kernel, spirit_kernels, m, n, no_c).cuda() - torch.reshape(x[0,...],(m*n*no_c,1))
            b2 = self.conv_backward_new(b1, kernel, spirit_kernels, m, n, no_c).cuda() - b1
            res1 = torch.reshape(a2 + rho*b2 ,(m,n,no_c))  #rho*b2
            #res = a2 + 1*b2   #rho*b2 #OPEN THIS LINE TO CANCEL REGULARIZATION
            return torch.unsqueeze(res1,dim=0)
        
        warm_start_kspace = prev_kspace[:,:,:,0:divp] + prev_kspace[:,:,:,divp:]*1j # turning real-imag to complex!
        p_now = DTy + torch.abs(self.miu)*(z[:,:,:,0:divp] + z[:,:,:,divp:]*1j) -  ATA(warm_start_kspace,kernel,spirit_kernels,locs,m,n,no_c) - torch.abs(self.miu)*warm_start_kspace 
        r_now = torch.clone( p_now)
        b_approx = warm_start_kspace 

        for i in range(10):
            
            q = ATA(p_now,kernel,spirit_kernels,locs,m,n,no_c) + torch.abs(self.miu)*p_now; 
            rrOverpq = torch.sum(r_now*torch.conj(r_now)) / torch.sum(q*torch.conj(p_now))  
            b_next = b_approx + rrOverpq*p_now
            r_next = r_now - rrOverpq*q;   
            p_next = r_next + torch.sum(r_next*torch.conj(r_next)) / torch.sum(r_now*torch.conj(r_now)) * p_now 
            b_approx = b_next
    
            p_now = torch.clone(p_next)
            r_now = torch.clone(r_next)

        return torch.cat([torch.real(b_approx), torch.imag(b_approx)], dim=3)
        
    
    def selection_operator(self,x, locations, m, n, no_c):
        x = torch.reshape(x,(m, n, no_c))
        y = x[locations]
        return y

    def adjoint_selection_operator(self,x, locations, m, n, no_c):
        space = torch.zeros((m, n, no_c), dtype=torch.complex64).cuda()
        
        space[locations] = x
        
        return torch.reshape(space,(m*n*no_c,1))

    def conv_forward_new(self,kspace, kernel, spirit_kernels, m, n, no_c):
        
        kernel = torch.permute(torch.reshape(kernel[0,...],(9, 9, no_c,no_c)),(0,1,2,3)).cuda()

        return torch.reshape(torch.permute(torch.squeeze(torch.nn.functional.conv2d(torch.permute(torch.reshape(kspace[...],(1,m,n,no_c)),(0,3,1,2)), torch.permute(torch.reshape(kernel[...],(9,9,no_c,no_c)),(3,2,1,0)),padding=4),0),(1,2,0)),(m*n*no_c,1))
        
    
    
    
    def conv_backward_new(self,kspace, kernel, spirit_kernels, m, n, no_c):
        
        kernel_mid = torch.zeros(9,9,no_c,no_c,dtype=torch.complex64).cuda()
        for ii in range(no_c):
            for jj in range(no_c):
                mid = torch.reshape(kernel[0,:,ii, jj],(9,9))
                kernel_mid[...,ii,jj] = torch.conj((torch.flip(mid,[1,0])))

        return torch.reshape(torch.permute(torch.squeeze(torch.nn.functional.conv2d(torch.permute(torch.reshape(kspace[...],(1,m,n,no_c)),(0,3,1,2)), torch.permute(kernel_mid[:,:,:,:],(2,3,1,0)),padding=4),0),(1,2,0)),(m*n*no_c,1))
        
    
    
        
    
