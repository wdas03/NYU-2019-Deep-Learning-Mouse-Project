import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sync_batchnorm.batchnorm import SynchronizedBatchNorm3d

class Decoder_ELU(nn.Module):
    def __init__(self, num_classes):
        super(Decoder_ELU, self).__init__()
        low_level_inplanes = 32
        
        self.conv1 = nn.Conv3d(low_level_inplanes, 12, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(12)
        self.elu = nn.ELU()
        
        self.transpose = nn.ConvTranspose3d(64, 64, kernel_size=16, stride=16)
        
        self.last_conv = nn.Sequential(nn.Conv3d(76, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm3d(64),
                                       nn.ELU(),
                                       nn.Dropout(0.5),
                                       nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm3d(64),
                                       nn.ELU(),
                                       nn.Dropout(0.1),
                                       nn.Conv3d(64, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.elu(low_level_feat)
        #print('low_level', low_level_feat.shape)
        
        #print('x', x.shape)
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='trilinear', align_corners=True)
        #x = self.transpose(x)
        #print(x.shape)
        x = torch.cat((x, low_level_feat), dim=1)
        #print(x.shape)
        x = self.last_conv(x)

        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.normal_(m.weight, mean=0, std=1)

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        low_level_inplanes = 32
        
        self.conv1 = nn.Conv3d(low_level_inplanes, 12, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(12)
        self.relu = nn.ReLU()
        
        self.transpose = nn.ConvTranspose3d(64, 64, kernel_size=16, stride=16)
        
        self.last_conv = nn.Sequential(nn.Conv3d(76, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm3d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm3d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv3d(64, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        #print('low_level', low_level_feat.shape)
        
        #print('x', x.shape)
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='trilinear', align_corners=True)
        #x = self.transpose(x)
        #print(x.shape)
        x = torch.cat((x, low_level_feat), dim=1)
        #print(x.shape)
        x = self.last_conv(x)

        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.normal_(m.weight, mean=0, std=1)