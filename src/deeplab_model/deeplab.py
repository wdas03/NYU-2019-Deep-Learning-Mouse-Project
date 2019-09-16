import torch
import torch.nn as nn
import torch.nn.functional as F
from .aspp import *
from .decoder import *
from .AlignedXception import *

from sync_batchnorm import convert_model

class DeepLab_ELU(nn.Module):
    def __init__(self, output_stride=8):
        super(DeepLab_ELU, self).__init__()

        self.backbone = AlignedXception_ELU(3)
        self.aspp = ASPP_ELU(output_stride) # changed from 2 to 8
        self.decoder = Decoder_ELU(3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        #print(x.shape)
        x = self.decoder(x, low_level_feat)
        #print(x.shape)
        x = F.interpolate(x, size=input.shape[2:], mode='trilinear', align_corners=True)
        
        x = self.softmax(x)
        #print(x.shape)
        return x

class DeepLabModified(nn.Module):
    def __init__(self, output_stride=8, elu=False):
        super(DeepLabModified, self).__init__()
        
        if elu:
            self.backbone = AlignedXception_ELU(3)
            self.aspp = ASPP_ELU(output_stride)
            self.decoder = Decoder_ELU(3)
        else:
            self.backbone = AlignedXception(3)
            self.aspp = ASPP(output_stride) # changed from 2 to 8
            self.decoder = Decoder(3)
        
        
        self.last_convs = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 3, kernel_size=3, stride=1, padding=1),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        #print(x.shape)
        x = self.decoder(x, low_level_feat)
        #print(x.shape)
        x = F.interpolate(x, size=input.shape[2:], mode='trilinear', align_corners=True)
        combine = torch.cat([x, input], dim=1)
        x = self.last_convs(combine)
        
        x = self.softmax(x)
        #print(x.shape)
        return x

class DeepLab(nn.Module):
    def __init__(self, output_stride=8):
        super(DeepLab, self).__init__()

        self.backbone = AlignedXception(3)
        self.aspp = ASPP(output_stride) # changed from 2 to 8
        self.decoder = Decoder(3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        #print(x.shape)
        x = self.decoder(x, low_level_feat)
        #print(x.shape)
        x = F.interpolate(x, size=input.shape[2:], mode='trilinear', align_corners=True)
        
        x = self.softmax(x)
        #print(x.shape)
        return x
