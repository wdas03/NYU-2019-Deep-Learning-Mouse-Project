# Import all necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from sync_batchnorm.batchnorm import SynchronizedBatchNorm3d

# Important classes: BatchNormRelu3D, BatchNorm3D
from data_utils import *

# Xception network adapted for 3D images
def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False):
        super(DepthwiseSeparableConv3D, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.bn = nn.BatchNorm3d(in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        
    def forward(self, x):
        #x = fixed_padding(x, self.depthwise.kernel_size[0], dilation=self.depthwise.dilation[0])
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class MiddleFlowBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiddleFlowBlock, self).__init__()
        self.sequenceOfConv = nn.Sequential(
            DepthwiseSeparableConv3D(in_channels, out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv3D(out_channels, out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv3D(out_channels, out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        residual = x
        x = self.sequenceOfConv(x)
        x = x + residual
        return F.relu(x)
    
# Modified Xception
class AlignedXception(nn.Module):
    def __init__(self, num_classes):
        super(AlignedXception, self).__init__()
        
        # Entry block
        self.first_two_conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        self.convBlock_1_residual = nn.Conv3d(16, 32, kernel_size=1, stride=2) # change stride
       
        self.separableConvBlock_1 = nn.Sequential(
            DepthwiseSeparableConv3D(16, 32),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv3D(32,32),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv3D(32, 32, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.convBlock_2_residual = nn.Conv3d(32, 64, kernel_size=1, stride=2) # changed stride
         
        self.separableConvBlock_2 = nn.Sequential(
            DepthwiseSeparableConv3D(32, 64),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv3D(64, 64),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv3D(64, 64, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.convBlock_3_residual = nn.Conv3d(64, 182, kernel_size=1, stride=2) # changed stride
        
        self.separableConvBlock_3 = nn.Sequential(
            DepthwiseSeparableConv3D(64, 182),
            nn.BatchNorm3d(182),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv3D(182, 182),
            nn.BatchNorm3d(182),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv3D(182, 182, stride=2),
            nn.BatchNorm3d(182),
            nn.ReLU(inplace=True)
        )
        
        # Middle block
        self.middleBlock_1 = MiddleFlowBlock(182, 182)
        self.middleBlock_2 = MiddleFlowBlock(182, 182)
        self.middleBlock_3 = MiddleFlowBlock(182, 182)
        self.middleBlock_4 = MiddleFlowBlock(182, 182)
        self.middleBlock_5 = MiddleFlowBlock(182, 182)
        self.middleBlock_6 = MiddleFlowBlock(182, 182)
        self.middleBlock_7 = MiddleFlowBlock(182, 182)
        self.middleBlock_8 = MiddleFlowBlock(182, 182)
        self.middleBlock_9 = MiddleFlowBlock(182, 182)
        self.middleBlock_10 = MiddleFlowBlock(182, 182)
        self.middleBlock_11 = MiddleFlowBlock(182, 182)
        self.middleBlock_12 = MiddleFlowBlock(182, 182)
        self.middleBlock_13 = MiddleFlowBlock(182, 182)
        self.middleBlock_14 = MiddleFlowBlock(182, 182)
        self.middleBlock_15 = MiddleFlowBlock(182, 182)
        self.middleBlock_16 = MiddleFlowBlock(182, 182)
        
        # Exit flow
        self.convBlock_4_residual = nn.Conv3d(182, 256, kernel_size=1, stride=2) # changed stride
        
        self.separableConvBlock_4 = nn.Sequential(
            DepthwiseSeparableConv3D(182, 182),
            nn.BatchNorm3d(182),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv3D(182, 256),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv3D(256, 256, stride=2),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        self.separableConvBlock_5 = nn.Sequential(
            DepthwiseSeparableConv3D(256, 384),
            nn.BatchNorm3d(384),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv3D(384, 384),
            nn.BatchNorm3d(384),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv3D(384, 512, stride=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )
        
        self._init_weight()
        
        
    def forward(self, x):
        # Entry flow
        x = self.first_two_conv(x)
        convBlock_1_residual = self.convBlock_1_residual(x)
        separableConvBlock_1 = self.separableConvBlock_1(x)
        separableSum_1 = F.relu(convBlock_1_residual + separableConvBlock_1)
        low_level_feature = separableSum_1
        
        convBlock_2_residual = self.convBlock_2_residual(separableSum_1)
        separableConvBlock_2 = self.separableConvBlock_2(separableSum_1)
        separableSum_2 = F.relu(convBlock_2_residual + separableConvBlock_2)
        
        convBlock_3_residual = self.convBlock_3_residual(separableSum_2)
        separableConvBlock_3 = self.separableConvBlock_3(separableSum_2)
        separableSum_3 = F.relu(convBlock_3_residual + separableConvBlock_3)
        
        # middle flow
        mid = self.middleBlock_1(separableSum_3)
        mid = self.middleBlock_2(mid)
        mid = self.middleBlock_3(mid)
        mid = self.middleBlock_4(mid)
        mid = self.middleBlock_5(mid)
        mid = self.middleBlock_6(mid)
        mid = self.middleBlock_7(mid)
        mid = self.middleBlock_8(mid)
        mid = self.middleBlock_9(mid)
        mid = self.middleBlock_10(mid)
        mid = self.middleBlock_11(mid)
        mid = self.middleBlock_12(mid)
        mid = self.middleBlock_13(mid)
        mid = self.middleBlock_14(mid)
        mid = self.middleBlock_15(mid)
        mid = self.middleBlock_16(mid)
        
        convBlock_4_residual = self.convBlock_4_residual(mid)
        separableConvBlock_4 = self.separableConvBlock_4(mid)
        separableSum_4 = F.relu(convBlock_4_residual + separableConvBlock_4)
        
        separableConvBlock_5 = self.separableConvBlock_5(separableSum_4)
        
        #print("low_level_feature:", low_level_feature.shape)
        #print("Xception output:", separableConvBlock_5.shape)
        
        return separableConvBlock_5, low_level_feature
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.normal_(m.weight, mean=0, std=1)

class MiddleFlowBlock_ELU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiddleFlowBlock_ELU, self).__init__()
        self.sequenceOfConv = nn.Sequential(
            DepthwiseSeparableConv3D(in_channels, out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ELU(inplace=True),
            
            DepthwiseSeparableConv3D(out_channels, out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ELU(inplace=True),
            
            DepthwiseSeparableConv3D(out_channels, out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ELU(inplace=True)
        )
    def forward(self, x):
        residual = x
        x = self.sequenceOfConv(x)
        x = x + residual
        return F.relu(x)
                
class AlignedXception_ELU(nn.Module):
    def __init__(self, num_classes):
        super(AlignedXception_ELU, self).__init__()
        
        # Entry block
        self.first_two_conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ELU(inplace=True),
            
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ELU(inplace=True)
        )
        
        self.convBlock_1_residual = nn.Conv3d(16, 32, kernel_size=1, stride=2) # change stride
       
        self.separableConvBlock_1 = nn.Sequential(
            DepthwiseSeparableConv3D(16, 32),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            
            DepthwiseSeparableConv3D(32,32),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            
            DepthwiseSeparableConv3D(32, 32, stride=2),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True)
        )
        
        self.convBlock_2_residual = nn.Conv3d(32, 64, kernel_size=1, stride=2) # changed stride
         
        self.separableConvBlock_2 = nn.Sequential(
            DepthwiseSeparableConv3D(32, 64),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            
            DepthwiseSeparableConv3D(64, 64),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            
            DepthwiseSeparableConv3D(64, 64, stride=2),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True)
        )
        
        self.convBlock_3_residual = nn.Conv3d(64, 182, kernel_size=1, stride=2) # changed stride
        
        self.separableConvBlock_3 = nn.Sequential(
            DepthwiseSeparableConv3D(64, 182),
            nn.BatchNorm3d(182),
            nn.ELU(inplace=True),
            
            DepthwiseSeparableConv3D(182, 182),
            nn.BatchNorm3d(182),
            nn.ELU(inplace=True),
            
            DepthwiseSeparableConv3D(182, 182, stride=2),
            nn.BatchNorm3d(182),
            nn.ELU(inplace=True)
        )
        
        # Middle block
        self.middleBlock_1 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_2 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_3 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_4 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_5 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_6 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_7 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_8 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_9 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_10 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_11 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_12 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_13 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_14 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_15 = MiddleFlowBlock_ELU(182, 182)
        self.middleBlock_16 = MiddleFlowBlock_ELU(182, 182)
        
        # Exit flow
        self.convBlock_4_residual = nn.Conv3d(182, 256, kernel_size=1, stride=2) # changed stride
        
        self.separableConvBlock_4 = nn.Sequential(
            DepthwiseSeparableConv3D(182, 182),
            nn.BatchNorm3d(182),
            nn.ELU(inplace=True),
            
            DepthwiseSeparableConv3D(182, 256),
            nn.BatchNorm3d(256),
            nn.ELU(inplace=True),
            
            DepthwiseSeparableConv3D(256, 256, stride=2),
            nn.BatchNorm3d(256),
            nn.ELU(inplace=True)
        )
        
        self.separableConvBlock_5 = nn.Sequential(
            DepthwiseSeparableConv3D(256, 384),
            nn.BatchNorm3d(384),
            nn.ELU(inplace=True),
            
            DepthwiseSeparableConv3D(384, 384),
            nn.BatchNorm3d(384),
            nn.ELU(inplace=True),
            
            DepthwiseSeparableConv3D(384, 512, stride=1),
            nn.BatchNorm3d(512),
            nn.ELU(inplace=True)
        )
        
        self._init_weight()
        
        
    def forward(self, x):
        # Entry flow
        x = self.first_two_conv(x)
        convBlock_1_residual = self.convBlock_1_residual(x)
        separableConvBlock_1 = self.separableConvBlock_1(x)
        separableSum_1 = F.relu(convBlock_1_residual + separableConvBlock_1)
        low_level_feature = separableSum_1
        
        convBlock_2_residual = self.convBlock_2_residual(separableSum_1)
        separableConvBlock_2 = self.separableConvBlock_2(separableSum_1)
        separableSum_2 = F.relu(convBlock_2_residual + separableConvBlock_2)
        
        convBlock_3_residual = self.convBlock_3_residual(separableSum_2)
        separableConvBlock_3 = self.separableConvBlock_3(separableSum_2)
        separableSum_3 = F.relu(convBlock_3_residual + separableConvBlock_3)
        
        # middle flow
        mid = self.middleBlock_1(separableSum_3)
        mid = self.middleBlock_2(mid)
        mid = self.middleBlock_3(mid)
        mid = self.middleBlock_4(mid)
        mid = self.middleBlock_5(mid)
        mid = self.middleBlock_6(mid)
        mid = self.middleBlock_7(mid)
        mid = self.middleBlock_8(mid)
        mid = self.middleBlock_9(mid)
        mid = self.middleBlock_10(mid)
        mid = self.middleBlock_11(mid)
        mid = self.middleBlock_12(mid)
        mid = self.middleBlock_13(mid)
        mid = self.middleBlock_14(mid)
        mid = self.middleBlock_15(mid)
        mid = self.middleBlock_16(mid)
        
        convBlock_4_residual = self.convBlock_4_residual(mid)
        separableConvBlock_4 = self.separableConvBlock_4(mid)
        separableSum_4 = F.relu(convBlock_4_residual + separableConvBlock_4)
        
        separableConvBlock_5 = self.separableConvBlock_5(separableSum_4)
        
        #print("low_level_feature:", low_level_feature.shape)
        #print("Xception output:", separableConvBlock_5.shape)
        
        return separableConvBlock_5, low_level_feature
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.normal_(m.weight, mean=0, std=1)
                
