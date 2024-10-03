# SCR-DL
Accelerated Cardiac Cine with Spatio-Coil Regularized Deep Learning Reconstruction

[[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3JWEBB)]

Please cite the following:

# SCR-DL

We sought to develop a spatio-coil regularized iterative deep learning reconstruction (SCR-DL) to accelerate cine CMR.

The trained model can be found under [[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3JWEBB)] with the name of "best_model_ResNet.pth".

Training and testing requires the following format as the input:

Training:
kspace = M x N x Coil (complex)
ref_kspace = M x N x Coil (complex)
mask = M x N x 2 (single) - this can be an empty matrix
acs_end = 1 - assign to 1
acs_start = 1 - assign to 1

Testing:
kspace = M x N x Coil (complex)
ref_kspace = M x N x Coil (complex)
mask = M x N x 2 (single) - this can be an empty matrix
acs_end = 1 - assign to 1
acs_start = 1 - assign to 1

An example dataset is included under [[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3JWEBB)] with the name of "example_test.mat". 

# Abstract
