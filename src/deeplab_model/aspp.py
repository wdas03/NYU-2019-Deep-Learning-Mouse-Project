import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sync_batchnorm.batchnorm import SynchronizedBatchNorm3d

class ASPP_ELU(nn.Module):
    def __init__(self, output_stride):
        super(ASPP_ELU, self).__init__()
        inplanes = 512
        
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        elif output_stride == 4:
            dilations = [1, 24, 48, 72]
        elif output_stride == 2:
            dilations = [1, 48, 96, 144]

        self.aspp1 = _ASPPModule_ELU(inplanes, 64, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule_ELU(inplanes, 64, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule_ELU(inplanes, 64, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule_ELU(inplanes, 64, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((16, 16, 16)),
                                             nn.Conv3d(inplanes, 64, 1, stride=1, bias=False),
                                             nn.BatchNorm3d(64),
                                             nn.ELU())
        self.conv1 = nn.Conv3d(320, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        #print('x4', x4.shape)
        x5 = self.global_avg_pool(x)
        #print('x5', x5.shape)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.normal_(m.weight, mean=0, std=1)

class _ASPPModule_ELU(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule_ELU, self).__init__()
        self.atrous_conv = nn.Conv3d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(planes)
        self.elu = nn.ELU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.elu(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.normal_(m.weight, mean=0, std=1)

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv3d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.normal_(m.weight, mean=0, std=1)

class ASPP(nn.Module):
    def __init__(self, output_stride):
        super(ASPP, self).__init__()
        inplanes = 512
        
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        elif output_stride == 4:
            dilations = [1, 24, 48, 72]
        elif output_stride == 2:
            dilations = [1, 48, 96, 144]

        self.aspp1 = _ASPPModule(inplanes, 64, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 64, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 64, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 64, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((16, 16, 16)),
                                             nn.Conv3d(inplanes, 64, 1, stride=1, bias=False),
                                             nn.BatchNorm3d(64),
                                             nn.ReLU())
        self.conv1 = nn.Conv3d(320, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        #print('x4', x4.shape)
        x5 = self.global_avg_pool(x)
        #print('x5', x5.shape)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.normal_(m.weight, mean=0, std=1)
