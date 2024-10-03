import torch
import torch.nn as nn
import torch.nn.functional as F
 
class ResidualBlock(nn.Module):
 
    # ResNet convs 
    def __init__(self, input_channels, output_channels, KernelSize=3, para_C=0.1):
        super().__init__()
        self.ConvPart = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=KernelSize, padding=(KernelSize//2, KernelSize//2, KernelSize//2),bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, kernel_size=KernelSize, padding=(KernelSize//2, KernelSize//2, KernelSize//2),bias=False),
        )
        self.C = para_C
 
 
    def forward(self, x):
        return x + self.ConvPart(x)*self.C
    



class BDResNet(nn.Module):
    
    def __init__(self, input_channels, intermediateChannels, output_channels, KernelSize=3):
        super().__init__()
        
        # First Layer mapping 2 channels to 64
        self.FirstLayer = nn.Conv3d(input_channels, intermediateChannels, kernel_size=KernelSize, padding='same',bias=False)
        # Then goes through residual blocks, all 64 channels 

        self.ResNetConv = nn.Sequential(
            ResidualBlock(intermediateChannels,intermediateChannels), # 1 
            ResidualBlock(intermediateChannels,intermediateChannels), # 2 
            ResidualBlock(intermediateChannels,intermediateChannels), # 3
            ResidualBlock(intermediateChannels,intermediateChannels), # 4 
            ResidualBlock(intermediateChannels,intermediateChannels), # 5 
            ResidualBlock(intermediateChannels,intermediateChannels), # 6
            ResidualBlock(intermediateChannels,intermediateChannels), # 7
            ResidualBlock(intermediateChannels,intermediateChannels), # 8
            ResidualBlock(intermediateChannels,intermediateChannels), # 9
            ResidualBlock(intermediateChannels,intermediateChannels), # 10
            ResidualBlock(intermediateChannels,intermediateChannels), # 11
            ResidualBlock(intermediateChannels,intermediateChannels), # 12
            ResidualBlock(intermediateChannels,intermediateChannels), # 13
            ResidualBlock(intermediateChannels,intermediateChannels), # 14
            ResidualBlock(intermediateChannels,intermediateChannels), # 15
            nn.Conv3d(intermediateChannels, intermediateChannels, kernel_size=KernelSize, padding=(KernelSize//2, KernelSize//2, KernelSize//2),bias=False)
            )
        
        # last layer is just convolution, with 64 to 2 channels as output.
        self.LastLayer = nn.Conv3d(intermediateChannels, output_channels, kernel_size=KernelSize, padding='same',bias=False)
 
    def forward(self, x):
        
        FirstLayerOutput = self.FirstLayer(x)
        return self.LastLayer(FirstLayerOutput + self.ResNetConv(FirstLayerOutput))


# testing, if correct it should give a network description
'''
if __name__ == '__main__':
    net = ZcResNet(input_channels=2, intermediateChannels= 64, output_channels=2)
    print(net)
    
    # check parameters 
    for name, param in net.named_parameters():
        print(name, param.size(), type(param))

'''