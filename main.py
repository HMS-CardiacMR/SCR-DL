import BDUnrolledNet_3D as UnrolledNetwork
import BDDataLoader as FeedData

from torch import optim
import torch.nn as nn
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import time
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def tic():
    #Homemade version of matlab tic and toc functions
        global startTime_for_tictoc
        startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time Training is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

def ZcMixedLoss(recon, label, reconim, labelim):
    loss1 = 0.5*torch.norm(recon-label,p=1)/torch.norm(label,p=1) + 0.5*torch.norm(recon-label,p=2)/torch.norm(label,p=2)
    loss2 = torch.norm(reconim-labelim,p=1)/torch.norm(labelim,p=1) 
    return loss1,loss2
    #return torch.norm(recon-label,p=2)/torch.norm(label,p=2)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Trainable_num = ',trainable_num)
    print('Total var num = ',total_num)
    return 1

def Training(network, device, image_path, epochs=1001, batch_size=1, LearningRate=1e-4): #1e-4 #LearningRate=0.000005

    CartesianData = FeedData.BDCartesianSSDULoader(datapath = image_path)
    Data_sampler = torch.utils.data.RandomSampler(CartesianData)
    data_loader = torch.utils.data.DataLoader(dataset=CartesianData,
                                               batch_size=batch_size, 
                                               #shuffle=True)  # here we use sampler to test distributed version
                                               sampler = Data_sampler)
    get_parameter_number(network)
    optimizer = optim.Adam(network.parameters(), lr=LearningRate)
    network.eval()
    LossFunction = ZcMixedLoss
    #LossFunction = torch.nn.L1Loss()
    # best loss set to inf as starting point
    best_loss = float('inf')
    loss_List = []
    loss_List1 = []
    loss_List2 = []
    
    # max old PE 240, RO 290
    ### DATA LOADING ################################################################ 
    data_size = 1915
    PE_size = 208
    RO_size = 262
    Ch_size = 8
    RO_loc = []
    PE_loc = []
    kspace_all = torch.zeros(size =(data_size,RO_size,PE_size,Ch_size*2),dtype=torch.float32)
    ref_kspace_all = torch.zeros(size =(data_size,RO_size,PE_size,Ch_size*2),dtype=torch.float32)
    mask_all = torch.zeros(size =(data_size,RO_size,PE_size,2),dtype=torch.float32)
    kernel_all = torch.zeros(size =(data_size,9*9,Ch_size,Ch_size),dtype=torch.complex64)
    acs_start = []
    acs_end = []
    #RO_loc = np.zeros((data_size,1))
    #PE_loc = np.zeros((data_size,1))        
    #pre-loading
    
    dloc = 0
    for acsp,kspace,kernel,spirit_kernels,mask,ref_kspace, fileName in data_loader:
        print(dloc)
        RO_loc.append(kspace.shape[1])
        PE_loc.append(kspace.shape[2])
        acs_start.append(acsp[0])
        acs_end.append(acsp[1])
        kspace_all[dloc,:kspace.shape[1],:kspace.shape[2],...] = kspace
        kernel_all[dloc,...] = kernel
        mask_all[dloc,:kspace.shape[1],:kspace.shape[2],...] =mask
        ref_kspace_all[dloc,:kspace.shape[1],:kspace.shape[2],...] =ref_kspace
        dloc = dloc+1
    ################################################################  
    

    # epoches
    for epoch in range(epochs):
        # start torch.nn.module's training mode
        network.train()
        loss_buff = []
        loss_buff1 = []
        loss_buff2 = []
        # for loop over batches

        #for kspace,kernel,spirit_kernels,mask,ref_kspace, fileName in data_loader:
        for ii in range(data_size): 
            tic()
            torch.cuda.synchronize()
            optimizer.zero_grad()
            
            # do zero_grad before every iteration
            
            kspace = kspace_all[ii,...].cuda()
            kspace = kspace[:RO_loc[ii],:PE_loc[ii],:]    
            kernel = kernel_all[ii,...].cuda()
            mask = mask_all[ii,...].cuda()
            mask = mask[:RO_loc[ii],:PE_loc[ii],:] 
            ref_kspace = ref_kspace_all[ii,...].cuda()
            ref_kspace = ref_kspace[:RO_loc[ii],:PE_loc[ii],:]   

            kspace = torch.unsqueeze(kspace, 0)
            kernel = torch.unsqueeze(kernel, 0)
            mask = torch.unsqueeze(mask, 0)
            ref_kspace = torch.unsqueeze(ref_kspace, 0)

            acs_points = []
            acs_points.append(acs_start[ii])
            acs_points.append(acs_end[ii])
            

            kspace = kspace.cuda()#image.to(device=device, dtype=torch.float32)
            kernel = kernel.cuda()#label.to(device=device, dtype=torch.complex64)
            ref_kspace = ref_kspace.cuda()#coil.to(device=device, dtype=torch.complex64)
            mask = mask.cuda()#image.to(device=device, dtype=torch.float32)
   
          
            recon = network(kspace, kernel, spirit_kernels, mask,acs_points)


            r1 = recon
            r2 = ref_kspace
            
            kspace = kspace[...,0:Ch_size] + kspace[...,Ch_size:]*1j     
            unknown_kspace = torch.logical_not(torch.abs(kspace)> 0)
            
            
            r1 = recon[...,0:Ch_size] + recon[...,Ch_size:]*1j
            r1 = r1[unknown_kspace]
            r1 = torch.cat([torch.real(r1), torch.imag(r1)], dim=0)

            r2 = ref_kspace[...,0:Ch_size] + ref_kspace[...,Ch_size:]*1j
            r2 = r2[unknown_kspace]
            r2 = torch.cat([torch.real(r2), torch.imag(r2)], dim=0)
            
            ##image domain
            recon = recon[...,0:Ch_size] + recon[...,Ch_size:]*1j
            dum = ref_kspace[...,0:Ch_size] + ref_kspace[...,Ch_size:]*1j  
            
            recon[torch.logical_not(unknown_kspace)] = dum[torch.logical_not(unknown_kspace)]
            
            reconim = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(recon,dim = (1,2)), dim=(1,2), norm = 'forward'),dim = (1,2))
            reconim = torch.cat([torch.real(reconim), torch.imag(reconim)], dim=3)

            refim = ref_kspace[...,0:Ch_size] + ref_kspace[...,Ch_size:]*1j
            refim = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(refim,dim = (1,2)), dim=(1,2), norm = 'forward'),dim = (1,2))
            refim = torch.cat([torch.real(refim), torch.imag(refim)], dim=3)

            # get loss
            l1,l2 = LossFunction(r1,r2,reconim,refim) 
            loss = l1+l2#LossFunction(r1,r2)#l1 #+ l2



            print('mu = ',network.DataConsistency.miu.cpu().detach().numpy(),'loss1 = ',l1.cpu().detach().numpy(),'___loss2 = ',l2.cpu().detach().numpy())
       
            loss_buff = np.append(loss_buff, loss.item())
            loss_buff1 = np.append(loss_buff1, l1.cpu().detach().numpy())
            loss_buff2 = np.append(loss_buff2, l2.cpu().detach().numpy())
            # backpropagate
            loss.backward()
        
            # update parameters
            optimizer.step()
            toc()


        if(epoch % 1==0):
             torch.save(network.state_dict(), 'define your data path/best_model_ResNet_Epoch%d.pth'%epoch)


        loss_List = np.append(loss_List, np.mean(loss_buff)/2)    
        loss_List1 = np.append(loss_List1, np.mean(loss_buff1)/2)    
        loss_List2 = np.append(loss_List2, np.mean(loss_buff2)/2)    
        sio.savemat('LossCurve.mat',{'loss':loss_List,'loss1':loss_List1,'loss2':loss_List2})
        print('$$$$$$$$$$$$$ Average Loss = ', np.mean(loss_buff)/2,', at epoch', epoch,'$$$$$$$$$$$$$$')
        if(epoch % 1==0):
            sio.savemat('ReconEpoch%d.mat'%epoch,{'recon':recon.cpu().detach().numpy(),'ref':ref_kspace.cpu().detach().numpy()})
        
        
    return recon, loss_List
 

if __name__ == "__main__":
    
    rate = 4
    
    # check CUDA availiability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    

    network = UnrolledNetwork.BDUnrolledNet(0.5,10) 
    # move to device
    network = network.cuda()#network.to(device=device)

    # data_path
    imageroute = 'define your data path'

    [recon,loss_List] = Training(network, device, imageroute )

