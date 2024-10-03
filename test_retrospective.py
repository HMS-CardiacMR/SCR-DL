
import BDUnrolledNet_3D as UnrolledNetwork
import BDDataLoader_test_retro as FeedData

from torch import optim
import torch.nn as nn
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
    retval = None
    print("hi1")
    print(optimizer_state)
    print("hi2")
    if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
        retval = optimizer.step(*args, **kwargs)
    return retval

def ZcMixedLoss(recon, label):
    return 0.5*torch.norm(recon-label,p=1)/torch.norm(label,p=1) + 0.5*torch.norm(recon-label,p=2)/torch.norm(label,p=2)
    #return torch.norm(recon-label,p=2)/torch.norm(label,p=2)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Trainable_num = ',trainable_num)
    print('Total var num = ',total_num)
    return 1


def Training(network, device, image_path,main_s, epochs=25, batch_size=1, LearningRate=1e-4): #3e-4 #LearningRate=0.000005
    
    get_parameter_number(network)
    optimizer = optim.Adam(network.parameters(), lr=LearningRate)

    LossFunction = ZcMixedLoss
    # best loss set to inf as starting point
    best_loss = float('inf')
    loss_List = []
    

    
    for epoch in range(epochs):
        network.eval()
        loss_buff = []
        # for loop over batches

        CartesianData = FeedData.ZcCartesianSSDULoader(datapath = image_path, sno=epoch+1)
        Data_sampler = torch.utils.data.RandomSampler(CartesianData)
        data_loader = torch.utils.data.DataLoader(dataset=CartesianData,
                                               batch_size=batch_size, 
                                               #shuffle=True)  # here we use sampler to test distributed version
                                               sampler = Data_sampler)

        for acsp,kspace,kernel,spirit_kernels,mask,ref_kspace, fileName in data_loader:
            torch.cuda.synchronize()
 

            acs_points = []
            acs_points.append(acsp[0])
            acs_points.append(acsp[1])

            kspace = kspace.cuda()#image.to(device=device, dtype=torch.float32)
            kernel = kernel.cuda()#label.to(device=device, dtype=torch.complex64)
            ref_kspace = ref_kspace.cuda()#coil.to(device=device, dtype=torch.complex64)
            mask = mask.cuda()#image.to(device=device, dtype=torch.float32)
            
            
            recon = network(kspace, kernel, spirit_kernels, mask,acs_points)
 
            # get loss
            loss = LossFunction(recon, ref_kspace) 
            print('mu = ',network.DataConsistency.miu.cpu().detach().numpy(),'loss = ',loss.cpu().detach().numpy())
 
            sio.savemat('/define your parth/test_subject'+str(main_s+1)+'/PROP_TEST%d.mat'%(epoch+1),{'recon':recon.cpu().detach().numpy(),'ref':ref_kspace.cpu().detach().numpy(),'kin':kspace.cpu().detach().numpy()})
            

            loss_buff = np.append(loss_buff, loss.item())


        loss_List = np.append(loss_List, np.mean(loss_buff)/2)    
        print('$$$$$$$$$$$$$ Average Loss = ', np.mean(loss_buff)/2,', at epoch', epoch,'$$$$$$$$$$$$$$')

    return recon, loss_List
 

if __name__ == "__main__":
    
    rate = 4
    
    # check CUDA availiability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    network = UnrolledNetwork.BDUnrolledNet(0.05,10)
        

    network.load_state_dict(torch.load("/define your path/best_model_ResNet.pth"))


    network.eval()
    # move to device
    network = network.cuda()#network.to(device=device)
    #network = network.to(device=device)


    for main_s in range(100):
        imageroute  ='/define your pathl/test_subject'+str(main_s+1)+'/'

        [recon,loss_List] = Training(network, device, imageroute,main_s )
