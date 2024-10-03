import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
#import ZcUnet2D as UNet
import BDResNet3D as ResNet3D
import BDDC as DC
import scipy.io as sio
import torch.nn.functional as F
import math
#from vit_pytorch import SimpleViT

class BDUnrolledNet(nn.Module):
    
    def __init__(self, miu, Unrolls):
        super().__init__()
 
        self.Network = ResNet3D.BDResNet(input_channels = 2*1, intermediateChannels = 64, output_channels = 2*1)
        self.DataConsistency = DC.BDDC_2D(miu)
        self.Unrolls = Unrolls
        self.miu = miu
 
    
    def UnrolledBlock(self,*args):
   
        
        midres = args[0]
        midres = midres[...,:8] + midres[...,8:]*1j
        midres = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(midres,dim = [1,2]), dim=[1,2], norm = 'ortho'),dim = [1,2]) #1,PE,RO,Ch complex
        midres = torch.cat([torch.real(midres), torch.imag(midres)], dim=3) #1,PE,RO,Chx2 real
  
        #### regularization step ######################
        midres = torch.permute(midres,(0,3,1,2))
        ##add for 3D conv
        midres = torch.unsqueeze(midres,dim=1)
        midres = torch.cat([midres[:,:,:8,...], midres[:,:,8:,...]], dim=1)    
        midres = self.Network( midres)
        ##remove for 3D conv
        midres = torch.cat([midres[:,0,...], midres[:,1,...]], dim=1) 
        midres = torch.squeeze(midres,dim=1) 
        midres = torch.permute(midres,(0,2,3,1)) #1,234,208,30

        midres = midres[...,:8] + midres[...,8:]*1j
        midres = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(midres,dim = [1,2]), dim=[1,2], norm = 'ortho'),dim = [1,2]) #1,PE,RO,Ch complex
        midres = torch.cat([torch.real(midres), torch.imag(midres)], dim=3) #1,PE,RO,Chx2 real
        #################################################
     

        return self.DataConsistency(midres, args[1], args[2] , args[3], midres)

   
    
    # here the forward step of unrolled network. 

    def forward(self, kspace, kernel, spirit_kernels, mask,acs_points):
        

        recon = self.DataConsistency( torch.zeros_like(kspace).cuda(), kspace, kernel, spirit_kernels, torch.zeros_like(kspace).cuda()) # Tikhonov

        it = 1
        for OuterIter in range(self.Unrolls):

            recon = checkpoint(self.UnrolledBlock, recon, kspace, kernel, spirit_kernels, mask,it,acs_points)
            it = it+1
        return recon


# testing, if correct it should give a network description
'''
if __name__ == '__main__':
    net = ZcUnrolledNet(1,0.001,5)
    print(net)
    # check parameters 
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
'''
