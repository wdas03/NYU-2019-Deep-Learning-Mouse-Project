import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import ConvRelu3D

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convgroup_1 = nn.Sequential(
            ConvRelu3D(1, 32, padding=1),
            ConvRelu3D(32, 32, padding=1)
        )
        self.maxpool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout3d(0.5)
        
        self.convgroup_2 = nn.Sequential(
            ConvRelu3D(32, 64, padding=1),
            ConvRelu3D(64, 64, padding=1)
        )
        
        self.maxpool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout3d(0.3)
        
        self.convgroup_3 = nn.Sequential(
            ConvRelu3D(64, 128, padding=1),
            ConvRelu3D(128, 128, padding=1)
        )
        
        self.maxpool_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout3d(0.3)
        
        self.convgroup_4 = ConvRelu3D(128, 128, padding=1)
        self.convgroup_5 = ConvRelu3D(128, 64, padding=1)
        self.convgroup_6 = ConvRelu3D(64, 32, padding=1)
        
        self.skip1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.skip2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        
        self.final_conv = nn.Conv3d(32, 3, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        convgroup_1 = self.convgroup_1(x)
        #print(convgroup_1.shape)
        maxpool_1 = self.drop1(self.maxpool_1(convgroup_1))
        convgroup_2 = self.convgroup_2(maxpool_1)
        #print(convgroup_2.shape)
        maxpool_2 = self.drop2(self.maxpool_2(convgroup_2))
        convgroup_3 = self.convgroup_3(maxpool_2)
        #print(convgroup_3.shape)
        maxpool_3 = self.drop3(self.maxpool_3(convgroup_3))
        convgroup_4 = self.convgroup_4(maxpool_3)
        #print(convgroup_4.shape)
        
        up1 = F.interpolate(convgroup_4, scale_factor=2, mode='trilinear', align_corners=True)
        skip1 = self.skip1(maxpool_2)
        add1 = up1 + skip1
        
        convgroup_5 = self.convgroup_5(add1)
        
        up2 = F.interpolate(convgroup_5, scale_factor=2, mode='trilinear', align_corners=True)
        skip2 = self.skip2(maxpool_1)
        add2 = up2 + skip2
        
        convgroup_6 = self.convgroup_6(add2)
        
        up3 = F.interpolate(convgroup_6, scale_factor=2, mode='trilinear', align_corners=True)
        final = self.final_conv(up3)
        out = self.softmax(final)
        
        return out