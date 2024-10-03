
import scipy.io as sio
import glob
from torch.utils.data import Dataset
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

class BDCartesianSSDULoader(Dataset):
    
    def __init__(self, datapath):
        self.data_path = glob.glob(os.path.join(datapath, 'subject*.mat'))

     
   
    def __getitem__(self, index):
        
        def tic():
        #Homemade version of matlab tic and toc functions
            global startTime_for_tictoc
            startTime_for_tictoc = time.time()

        def toc():
            if 'startTime_for_tictoc' in globals():
                print("Elapsed time DataRead is " + str(time.time() - startTime_for_tictoc) + " seconds.")
            else:
                print("Toc: start time not set")
        acsp = []
        # load k-space data
        # before tranpose max RO = 240, max PE = 290
        kspace = sio.loadmat(self.data_path[index])['kspace'].transpose([1,0,2])
        ref_kspace = sio.loadmat(self.data_path[index])['ref_kspace'].transpose([1,0,2])
        mask = sio.loadmat(self.data_path[index])['mask'].transpose([1,0,2])
        acsp.append(sio.loadmat(self.data_path[index])['acs_start'])
        acsp.append(sio.loadmat(self.data_path[index])['acs_end'])
        ##################################### SPIRIT Pre-processing for weights ############################################################
        tic()
        #hyperparameters
        spirit_kernels = (9,9) #SPIRIT kernel size, similar to GRAPPA
        acs_filt = 12 #means 16x2 x 16x2 acs region
        reg = 1e-2 #spirit weight regularization

        #extract the ACS region by 16x16. FOr performance improvement you can take the fully sampled RO so 16xRO, but this will increase computation time
        [PE,RO,no_c] = kspace.shape
        #acs = kspace[PE//2 -1 -acs_filt:PE//2 -1 +acs_filt,RO//2 -1 -acs_filt:RO//2 -1 +acs_filt,:]
        acs = kspace[PE//2 -1 -acs_filt:PE//2 -1 +acs_filt,:,:]
        #sio.savemat('acs.mat',{'acs':acs})
        
        kernel_dim = spirit_kernels[0] * spirit_kernels[0]
        MA = np.zeros(((acs.shape[0] - (spirit_kernels[0] - 1)) * (acs.shape[1] - (spirit_kernels[1] - 1)), kernel_dim * acs.shape[2])).astype(complex)

        # MA matrix filling
        # Kernels are shifted over the ACS region to form the calibration matrix
        for coil_selec in range(acs.shape[2]):
            selected_acs = acs[:, :, coil_selec]
            row_count = 0  # Changed to 0 since Python uses 0-based indexing
            for col in range(selected_acs.shape[1] - (spirit_kernels[1] - 1)):
                for row in range(selected_acs.shape[0] - (spirit_kernels[0] - 1)):
                    neighbors = selected_acs[row:row + spirit_kernels[0], col:col + spirit_kernels[1]]
                    neighbors = np.reshape(neighbors,(neighbors.shape[0]*neighbors.shape[1]),order='F')  # Flatten the neighbors array
                    MA[row_count, (coil_selec * kernel_dim):(coil_selec * kernel_dim + kernel_dim)] = neighbors
                    row_count += 1

        row_start = (spirit_kernels[0] // 2)
        row_end = acs.shape[0] - (spirit_kernels[0] // 2)
        col_start = (spirit_kernels[1] // 2)
        col_end = acs.shape[1] - (spirit_kernels[1] // 2)
        # Corresponding middle points of the kernels are formed into destination vectors.
        Mk = np.zeros((MA.shape[0], acs.shape[2])).astype(complex)

        # Mk vectors filling
        for coil_selec in range(acs.shape[2]):
            selected_acs = acs[row_start:row_end, col_start:col_end, coil_selec]
            Mk[:, coil_selec] = np.reshape(selected_acs,(selected_acs.shape[0]*selected_acs.shape[1]),order='F')

        ak = np.zeros((kernel_dim * acs.shape[2] - 1, acs.shape[2])).astype(complex)

        #Mk is the destianation points
        #MA is the what if we shift kernel all over the acs
        #ak is what we want to have in the format of 5x5xChxCh 

        # Pre-calculation of the matrix multiplications
        # To not calculate it every time, gives the format of Ax=b where x is the ak
        A = np.dot(MA.conj().T, MA)
        B = np.dot(MA.conj().T, Mk)

        #SPIRiT needs ch=1, middle 5x5 grid to equal to zero when it is projected to ch=1. The rest is normal like in GRAPPA
        #Therefore, we need to decrease A,B by 1 size each direction and calculate the lsq
        lambda_value = np.linalg.norm(A, 'fro') / np.sqrt(A.shape[0]) * reg
        print('SPIRiT reg constant: ' + str(lambda_value))
        for coil_selec in range(no_c):
            #print(coil_selec)
            A1 = A[(0):((kernel_dim//2)) + (coil_selec*kernel_dim), :]
            A2 = A[((kernel_dim//2) +1+ (coil_selec*kernel_dim)):, :]
            newA = np.concatenate((A1, A2), axis=0)
            A3 = newA[:, (0):((kernel_dim//2)) + (coil_selec*kernel_dim)]
            A4 = newA[:, ((kernel_dim//2)+1+ (coil_selec*kernel_dim)):]
            newA = np.concatenate((A3, A4), axis=1)

            B1 = B[(0):((kernel_dim//2)) + (coil_selec*kernel_dim), :]
            B2 = B[((kernel_dim//2) +1+ (coil_selec*kernel_dim)):, :]
            newB = np.concatenate((B1, B2), axis=0)
            

            ak[:, coil_selec] = np.linalg.inv(newA + np.eye(newA.shape[0])*lambda_value).dot(newB[:, coil_selec])
            #ak[:, coil_selec] = np.linalg.lstsq(newA + np.eye(newA.shape[0]) * lambda_value, newB[:, coil_selec])[0]

        new_ak = np.zeros((A.shape[0],no_c)).astype(np.complex64)

        #reshapeing back to original size by adding the zero to ch=1 when it sess ch=1
        for coil_selec in range(no_c):
            p1 = ak[0:(kernel_dim//2) + (coil_selec)*kernel_dim , coil_selec]
            p2 = ak[(kernel_dim//2) + (coil_selec)*kernel_dim:, coil_selec]
            new_ak[:, coil_selec] = np.concatenate((p1, [0], p2))
        ak = new_ak
        kernel = np.reshape(new_ak,(kernel_dim,no_c,no_c),order='F')

        kspace = (np.concatenate([np.real(kspace), np.imag(kspace)], axis=2))
        ref_kspace = (np.concatenate([np.real(ref_kspace), np.imag(ref_kspace)], axis=2))
        toc()
        return acsp,kspace,kernel,spirit_kernels,mask,ref_kspace, self.data_path[index] 
       
        
    def __len__(self):
        return len(self.data_path)
    
    # Copied from BD's code
    def GetGaussianMask(self, rho=0.4, num_iter=0):
        [nx, ny] = self.SamplingMask.shape
        count = 0
        test_pts = np.int(np.ceil(np.sum(self.SamplingMask[:]) * rho))
        Mask_Validation = np.zeros_like(self.SamplingMask)
        temp_mask = np.copy(self.SamplingMask)
        mx = self.SamplingMask.shape[0]//2
        my = self.SamplingMask.shape[1]//2
        #if num_iter == 0:
            #print('center of kspace, mx: ', mx, ', my: ', my)
        temp_mask[mx - 2: mx + 2, my - 2: my + 2] = 0
        while count <= test_pts:
            indx = np.int(np.round(np.random.normal(loc=mx, scale=(nx - 1) / 2)))
            indy = np.int(np.round(np.random.normal(loc=my, scale=(ny - 1) / 2)))
            if (0 <= indx < nx and 0 <= indy < ny and temp_mask[indx, indy] == 1 and Mask_Validation[indx, indy] != 1):
                Mask_Validation[indx, indy] = 1
                count = count + 1
        Mask_Training = self.SamplingMask - Mask_Validation
        
        return np.complex64(Mask_Training), np.complex64(Mask_Validation)
    
    