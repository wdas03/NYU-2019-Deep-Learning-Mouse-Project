import torch
import torch.nn as nn
import torch.nn.functional as F

class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 4, kernel_size=7, padding=3, stride=1)
   
        self.group1 = ResidualGroup(4, 8, stride=1)
        
        self.group2 = ResidualGroup(8, 16, stride=2)
        self.pdc_2 = PyramidDilatedConv(16, 16, stride=4, kernel_size=3, dconv_filters=16)
        
        self.group3 = ResidualGroup(16, 32, stride=2)
        self.pdc_3 = PyramidDilatedConv(32, 32, stride=2, kernel_size=3, dconv_filters=32)
        
        self.group4 = ResidualGroup(32, 64, stride=2)
        self.pdc_4 = PyramidDilatedConv(64, 64, stride=1, kernel_size=1, dconv_filters=64)
        
        self.group5 = ResidualGroup(64, 128, stride=1)
        self.pdc_5 = PyramidDilatedConv(128, 64, stride=1, kernel_size=1, dconv_filters=64)
        
        self.dropout = nn.Dropout3d(0.5)
        
        self.final_conv = nn.Conv3d(704, 3, kernel_size=1)
        self.final_softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        
        group1 = self.group1(x)
        
        group2 = self.group2(group1)
        #print(group2.shape)
        out_pdc1 = self.pdc_2(group2)
        #print("out_pdc1:", out_pdc1.shape)
        
        group3 = self.group3(group2)
        #print("group3:", group3.shape)
        out_pdc2 = self.pdc_3(group3)
        #print("out_pdc2:", out_pdc2.shape)
        
        group4 = self.group4(group3)
        #print(group4.shape)
        out_pdc3 = self.pdc_4(group4)
        #print("out_pdc3:", out_pdc3.shape)
        
        group5 = self.group5(group4)
        out_pdc4 = self.pdc_5(group5)
        #print("out_pdc4:", out_pdc4.shape)
        
        final_concat = torch.cat([out_pdc1, out_pdc2, out_pdc3, out_pdc4], dim=1)
        
        dropout = self.dropout(final_concat)
        dropout = F.interpolate(dropout, scale_factor=4, mode='trilinear', align_corners=True)
        final_conv = self.final_conv(dropout)
        final_softmax = self.final_softmax(final_conv)
        #print(final_softmax.shape)
        return final_softmax
        
class ResidualGroup(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(ResidualGroup, self).__init__()
        self.block1 = ResidualBlock(in_ch, out_ch, stride=stride)
        self.block2 = ResidualBlock(out_ch, out_ch, stride=1)
        self.block3 = ResidualBlock(out_ch, out_ch, stride=1)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
        
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(ResidualBlock, self).__init__()
        
        bottleN = int(out_ch / 4)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True)
        )
        
        self.conv1 = nn.Conv3d(in_ch, bottleN, kernel_size=1, stride=stride)
        self.bn_relu_1 = nn.Sequential(
            nn.BatchNorm3d(bottleN),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Conv3d(bottleN, bottleN, kernel_size=3)
        self.bn_relu_2 = nn.Sequential(
            nn.BatchNorm3d(bottleN),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv3d(bottleN, out_ch, kernel_size=1, padding=1)
        self.identity_map = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride)
        
    def forward(self, x):
        identity_map = self.bn_relu(x)
        
        conv1 = self.conv1(identity_map)
        conv1_bn = self.bn_relu_1(conv1)
        
        conv2 = self.conv2(conv1_bn)
        conv2_bn = self.bn_relu_2(conv2)
        
        final_conv = self.final_conv(conv2_bn)
        identity_map = self.identity_map(identity_map)
        
        output = identity_map + final_conv
        return output

class PyramidDilatedConv(nn.Module):
    def __init__(self, in_ch, number_kernel, stride, kernel_size, dconv_filters):
        super(PyramidDilatedConv, self).__init__()
        
        if kernel_size == 1:
            padding=0
        elif kernel_size == 3:
            padding = 1
        
        self.bn_relu = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True)
        )
        
        self.conv1 = nn.Conv3d(in_ch, number_kernel, kernel_size, stride=stride, padding=padding)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.a1 = nn.Sequential(
            nn.Conv3d(number_kernel, dconv_filters, kernel_size=1, dilation=1), 
            nn.BatchNorm3d(dconv_filters),
            nn.ReLU(inplace=True)
        )
        
        self.a2 = nn.Sequential(
            nn.Conv3d(number_kernel, dconv_filters, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm3d(dconv_filters),
            nn.ReLU(inplace=True)
        )
        
        self.a3 = nn.Sequential(
            nn.Conv3d(number_kernel, dconv_filters, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm3d(dconv_filters),
            nn.ReLU(inplace=True)
        )
        
        self.a4 = nn.Sequential(
            nn.Conv3d(number_kernel, dconv_filters, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm3d(dconv_filters),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        l = self.bn_relu(x)
        
        conv1 = self.relu1(self.conv1(l))
        
        a1 = self.a1(conv1)
        #print('a1:', a1.shape)
        a2 = self.a2(conv1)
        #print('a2:', a2.shape)
        a3 = self.a3(conv1)
        #print('a3:', a3.shape)
        a4 = self.a4(conv1)
        #print('a4:', a4.shape)
        concat = torch.cat([a1, a2, a3, a4], dim=1)
        
        return concat
        