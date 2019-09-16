"""
# Import all necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

<<<<<<< HEAD
import numpy as np

# Important classes: BatchNormRelu3D, BatchNorm3D
from data_utils import *


# TODO: Increase maximum output size to 128 cubed, rather than the current 64 cubed tensor
# change upsample from 2 to 4 and delete some layers
# ICNet - 3D Implementation: Image-Cascade Network
    
class ICNet(nn.Module):
    def __init__(self, num_classes):
        super(ICNet, self).__init__()
=======
# Important classes: BatchNormRelu3D, BatchNorm3D
from data_utils import *

# TODO: Increase maximum output size to 128 cubed, rather than the current 64 cubed tensor
# change upsample from 2 to 4 and delete some layers
# ICNet - 3D Implementation: Image-Cascade Network
class OriginalICNet(nn.Module):
    def __init__(self, num_classes):
        super(OriginalICNet, self).__init__()
>>>>>>> master
        # Initialize instance variables
        self.num_classes = num_classes
        
        #Conv3d function: 
<<<<<<< HEAD
        
        #torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,   padding_mode='zeros')
        
       # - More Notes on the input variables:
       #      - stride controls the stride for the cross-correlation.
       #      - padding controls the amount of implicit zero-paddings on both sides for padding number of points for each  dimension.
       #      - dilation controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this link has a nice visualization of what dilation does.
       #     - groups controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups. 
       #         - At groups=1, all inputs are convolved to all outputs.
       #         - At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels, and producing half the output channels, and both subsequently concatenated. 
       #         - At groups= in_channels, each input channel is convolved with its own set of filters, of size [out_channels/in_channels]
        
=======
        
        #torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,   padding_mode='zeros')
        
       # - More Notes on the input variables:
       #      - stride controls the stride for the cross-correlation.
       #      - padding controls the amount of implicit zero-paddings on both sides for padding number of points for each  dimension.
       #      - dilation controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this link has a nice visualization of what dilation does.
       #     - groups controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups. 
       #         - At groups=1, all inputs are convolved to all outputs.
       #         - At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels, and producing half the output channels, and both subsequently concatenated. 
       #         - At groups= in_channels, each input channel is convolved with its own set of filters, of size [out_channels/in_channels]
        
>>>>>>> master
        
        # BatchNormRelu3D and BatchNorm3D(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=False)
        
        # Convolutions for 1/2 Resolution of Image
        
        # Do trilinear interpolation to get 1/2 resolution
        # s2 = stride 2
        self.conv1_1_3x3_s2 = BatchNormRelu3D(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv1_2_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv1_3_3x3 = BatchNormRelu3D(8, 16,  3, 1, padding=1)
        
        # max pooling does not affect the number of out channels, but rather reduces the spatial dimensions of the output for simplicity in the model
        self.pool1_3x3_s2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Apply these convolutions to the max pooling layer
        self.conv2_1_1x1_proj = BatchNorm3D(16, 32, 1, 1)
        
        #self.conv2_1_bottleneck = BottleNeck(128, 32, 1) 
        # bottleneck
        self.conv2_1_1x1_reduce = BatchNormRelu3D(16, 8, 1, 1)
        self.conv2_1_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv2_1_1x1_increase = BatchNorm3D(8, 32, 1, 1)
        
        # conv2_2
<<<<<<< HEAD
        self.conv2_2_bottleneck = BottleNeck(32, 8, dilation=1) # bottleneck
        #self.conv2_2_1x1_reduce = BatchNormRelu3D(32, 8, 1, 1)
        #self.conv2_2_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        #self.conv2_2_1x1_increase = BatchNorm3D(8, 32, 1, 1)
        
        # conv2_3
        self.conv2_3_bottleneck = BottleNeck(32, 8, dilation=1)
        #self.conv2_3_1x1_reduce = BatchNormRelu3D(32, 8, 1, 1)
        #self.conv2_3_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        #self.conv2_3_1x1_increase = BatchNorm3D(8, 32, 1, 1)
=======
        #self.conv2_2_bottleneck = BottleNeck(32, 8, dilation=1) # bottleneck
        self.conv2_2_1x1_reduce = BatchNormRelu3D(32, 8, 1, 1)
        self.conv2_2_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv2_2_1x1_increase = BatchNorm3D(8, 32, 1, 1)
        
        # conv2_3
        #self.conv2_3_bottleneck = BottleNeck(32, 8, dilation=1)
        self.conv2_3_1x1_reduce = BatchNormRelu3D(32, 8, 1, 1)
        self.conv2_3_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv2_3_1x1_increase = BatchNorm3D(8, 32, 1, 1)
>>>>>>> master
        
        # conv3_1
        self.conv3_1_1x1_proj = BatchNorm3D(32, 64, 1, stride=2) # stride 2
        
        # bottleneck building block
        self.conv3_1_1x1_reduce = BatchNormRelu3D(32, 16, 1, 2) # stride 2
        self.conv3_1_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_1_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # self.conv3_1_bottleneck = BottleNeckConv3_1(256, 64, 1)
        
        # Convolutions for 1/4 Image Resolution
        
        # conv3_2 - bottleneck building block
<<<<<<< HEAD
        self.conv3_2_bottleneck = BottleNeck(64, 16, dilation=1)
        #self.conv3_2_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        #self.conv3_2_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        #self.conv3_2_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv3_3 - bottleneck building block
        self.conv3_3_bottleneck = BottleNeck(64, 16, dilation=1)
        #self.conv3_3_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        #self.conv3_3_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        #self.conv3_3_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv3_4
        self.conv3_4_bottleneck = BottleNeck(64, 16, dilation=1)
        #self.conv3_4_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        #self.conv3_4_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        #self.conv3_4_1x1_increase = BatchNorm3D(16, 64, 1, 1)
=======
        #self.conv3_2_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_2_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv3_2_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_2_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv3_3 - bottleneck building block
        #self.conv3_3_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_3_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv3_3_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_3_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv3_4
        #self.conv3_4_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_4_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv3_4_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_4_1x1_increase = BatchNorm3D(16, 64, 1, 1)
>>>>>>> master
        
        # conv4_1
        self.conv4_1_1x1_proj = BatchNorm3D(64, 128, 1, 1)
        
        #self.conv4_1_bottleneck = BottleNeck(512, 128, dilation=2)
        self.conv4_1_1x1_reduce = BatchNormRelu3D(64, 32, 1, 1)
        self.conv4_1_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_1_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_2
<<<<<<< HEAD
        self.conv4_2_bottleneck = BottleNeck(128, 32, dilation=2)
        #self.conv4_2_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        #self.conv4_2_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        #self.conv4_2_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_3
        self.conv4_3_bottleneck = BottleNeck(128, 32, dilation=2)
        #self.conv4_3_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        #self.conv4_3_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        #self.conv4_3_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_4
        self.conv4_4_bottleneck = BottleNeck(128, 32, dilation=2)
        #self.conv4_4_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        #self.conv4_4_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        #self.conv4_4_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_5
        self.conv4_5_bottleneck = BottleNeck(128, 32, dilation=2)
        #self.conv4_5_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        #self.conv4_5_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        #self.conv4_5_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_6
        self.conv4_6_bottleneck = BottleNeck(128, 32, dilation=2)
        #self.conv4_6_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        #self.conv4_6_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        #self.conv4_6_1x1_increase = BatchNorm3D(32, 128, 1, 1)
=======
        #self.conv4_2_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_2_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_2_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_2_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_3
        #self.conv4_3_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_3_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_3_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_3_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_4
        #self.conv4_4_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_4_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_4_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_4_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_5
        #self.conv4_5_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_5_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_5_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_5_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_6
        #self.conv4_6_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_6_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_6_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_6_1x1_increase = BatchNorm3D(32, 128, 1, 1)
>>>>>>> master
    
        # conv5_1
        self.conv5_1_1x1_proj = BatchNorm3D(128, 256, 1, 1)
        
        # bottleneck conv5-1
        self.conv5_1_1x1_reduce = BatchNormRelu3D(128, 64, 1, 1)
        self.conv5_1_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_1_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv5_2
<<<<<<< HEAD
        self.conv5_2_bottleneck = BottleNeck(256, 64, dilation=4)
        #self.conv5_2_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        #self.conv5_2_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        #self.conv5_2_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv5_3
        self.conv5_3_bottleneck = BottleNeck(256, 64, dilation=4)
        #self.conv5_3_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        #self.conv5_3_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        #self.conv5_3_1x1_increase = BatchNorm3D(64, 256, 1, 1)
=======
        #self.conv5_2_bottleneck = BottleNeck(256, 64, dilation=4)
        self.conv5_2_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv5_2_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_2_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv5_3
        #self.conv5_3_bottleneck = BottleNeck(256, 64, dilation=4)
        self.conv5_3_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv5_3_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_3_1x1_increase = BatchNorm3D(64, 256, 1, 1)
>>>>>>> master
        
        # Global average pooling here
        
        # conv5_3_sum
        self.conv5_4_k1 = BatchNorm3D(256, 64, 1, 1)
        self.conv_sub4 = BatchNorm3D(64, 32, 3, 1, padding=2, dilation=2) # dilation rate = 2
        self.conv3_1_sub2_proj = BatchNorm3D(64, 32, 1, 1)
        self.conv_sub2 = BatchNorm3D(32, 32, 3, 1, padding=2, dilation=2) # dilation rate = 2
        
        # Convolutions for 1 image resolution
        self.conv1_sub1 = BatchNormRelu3D(1, 8, 3, stride=2, padding=1) # stride = 2
        self.conv2_sub1 = BatchNormRelu3D(8, 8, 3, stride=2, padding=1) # stride = 2
        self.conv3_sub1 = BatchNormRelu3D(8, 16, 3, stride=2, padding=1) # stride = 2
        self.conv3_sub1_proj = BatchNorm3D(16, 32, 1, 1)
        
        self.aux_1_conv = nn.Conv3d(64, self.num_classes, 1, 1)
        self.aux_2_conv = nn.Conv3d(32, self.num_classes, 1, 1)
        self.classification = nn.Conv3d(32, self.num_classes, 1, 1)
        
        self.sigmoid = nn.Softmax(dim=1)
        self.aux_1_sigmoid = nn.Softmax(dim=1)
        self.aux_2_sigmoid = nn.Softmax(dim=1)
        
        # define loss function (self.loss = ...)
        
    # Input x: (shape = (frames, height, width))
    def forward(self, x):
        #print(x.shape)
        l, h, w = x.shape[2:] 

        # image resolution: 1/2 - L, H, W -> L/2, H/2, W/2
        data_sub2 = F.interpolate(x, scale_factor = 1/2, mode="trilinear", align_corners=True)

        # L/2, H/2, W/2 -> L/4, H/4, W/4
        conv1_1_3x3_s2 = self.conv1_1_3x3_s2(data_sub2)
        conv1_2_3x3 = self.conv1_2_3x3(conv1_1_3x3_s2)
        conv1_3_3x3 = self.conv1_3_3x3(conv1_2_3x3)

        # L/4, H/4, W/4 -> L/8, H/8, W/8
        pool1_3x3_s2 = self.pool1_3x3_s2(conv1_3_3x3)

        # L/8, H/8, W/8 -> L/16, H/16, W/16
        conv2_1_1x1_proj = self.conv2_1_1x1_proj(pool1_3x3_s2)
        
        # bottleneck
        conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_3x3 = self.conv2_1_3x3(conv2_1_1x1_reduce)
        conv2_1_1x1_increase = self.conv2_1_1x1_increase(conv2_1_3x3)
        
        # Add residual of conv2_1_1x1 projection to bottleneck of conv2_1
        conv2_1 = F.relu(conv2_1_1x1_proj + conv2_1_1x1_increase)
<<<<<<< HEAD
        conv2_2 = self.conv2_2_bottleneck(conv2_1)
        conv2_3 = self.conv2_3_bottleneck(conv2_2)
=======
        conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(conv2_1)
        conv2_2_3x3 = self.conv2_2_3x3(conv2_2_1x1_reduce)
        conv2_2_1x1_increase = self.conv2_2_1x1_increase(conv2_2_3x3)
        
        conv2_2 = F.relu(conv2_1 + conv2_2_1x1_increase)
        conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(conv2_2)
        conv2_3_3x3 = self.conv2_3_3x3(conv2_3_1x1_reduce)
        conv2_3_1x1_increase = self.conv2_3_1x1_increase(conv2_3_3x3)
        
        conv2_3 = F.relu(conv2_2 + conv2_3_1x1_increase)
>>>>>>> master
        
        conv3_1_1x1_proj = self.conv3_1_1x1_proj(conv2_3)
        
        # bottleneck residual block on conv2_3
        conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(conv2_3)
        conv3_1_3x3 = self.conv3_1_3x3(conv3_1_1x1_reduce)
        conv3_1_1x1_increase = self.conv3_1_1x1_increase(conv3_1_3x3)
        
        #conv3_1 = conv3_1_1x1_proj + conv3_1_1x1_increase
        conv3_1 = F.relu(conv3_1_1x1_proj + conv3_1_1x1_increase)
        
        # image resolution: 1/4 - L/16, H/16, W/16 -> L/32, H/32, W/32
        conv3_1_sub4 = F.interpolate(conv3_1, scale_factor=1/2, mode="trilinear", align_corners=True)
        
        #print("conv3_1 - x_sub4 shape:", x_sub4.shape)
        
<<<<<<< HEAD
        conv3_2 = self.conv3_2_bottleneck(conv3_1_sub4)
        conv3_3 = self.conv3_3_bottleneck(conv3_2)
        conv3_4 = self.conv3_4_bottleneck(conv3_3)
=======
        conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(conv3_1_sub4)
        conv3_2_3x3 = self.conv3_2_3x3(conv3_2_1x1_reduce)
        conv3_2_1x1_increase = self.conv3_2_1x1_increase(conv3_2_3x3)
        
        conv3_2 = F.relu(conv3_1_sub4 + conv3_2_1x1_increase)
        conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(conv3_2)
        conv3_3_3x3 = self.conv3_3_3x3(conv3_3_1x1_reduce)
        conv3_3_1x1_increase = self.conv3_3_1x1_increase(conv3_3_3x3)
        
        conv3_3 = F.relu(conv3_2 + conv3_3_1x1_increase)
        conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(conv3_3)
        conv3_4_3x3 = self.conv3_4_3x3(conv3_4_1x1_reduce)
        conv3_4_1x1_increase = self.conv3_4_1x1_increase(conv3_4_3x3)
        
        conv3_4 = F.relu(conv3_3 + conv3_4_1x1_increase)
>>>>>>> master
        
        # increase projection
        conv4_1_1x1_proj = self.conv4_1_1x1_proj(conv3_4)
        
        # bottlenecks
        conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(conv3_4)
        conv4_1_3x3 = self.conv4_1_3x3(conv4_1_1x1_reduce)
        conv4_1_1x1_increase = self.conv4_1_1x1_increase(conv4_1_3x3)
        
        conv4_1 = F.relu(conv4_1_1x1_proj + conv4_1_1x1_increase)
        #conv4_1 = conv4_1_1x1_proj + conv4_1_1x1_increase
<<<<<<< HEAD
=======
        conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(conv4_1)
        conv4_2_3x3 = self.conv4_2_3x3(conv4_2_1x1_reduce)
        conv4_2_1x1_increase = self.conv4_2_1x1_increase(conv4_2_3x3)
        
        conv4_2 = F.relu(conv4_1 + conv4_2_1x1_increase)
        conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(conv4_2)
        conv4_3_3x3 = self.conv4_3_3x3(conv4_3_1x1_reduce)
        conv4_3_1x1_increase = self.conv4_3_1x1_increase(conv4_3_3x3)
>>>>>>> master
        
        conv4_3 = F.relu(conv4_2 + conv4_3_1x1_increase)
        conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(conv4_3)
        conv4_4_3x3 = self.conv4_4_3x3(conv4_4_1x1_reduce)
        conv4_4_1x1_increase = self.conv4_4_1x1_increase(conv4_4_3x3)
        
        conv4_4 = F.relu(conv4_3 + conv4_4_1x1_increase)
        conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(conv4_4)
        conv4_5_3x3 = self.conv4_5_3x3(conv4_5_1x1_reduce)
        conv4_5_1x1_increase = self.conv4_5_1x1_increase(conv4_5_3x3)
        
        conv4_5 = F.relu(conv4_4 + conv4_5_1x1_increase)
        conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(conv4_5)
        conv4_6_3x3 = self.conv4_6_3x3(conv4_6_1x1_reduce)
        conv4_6_1x1_increase = self.conv4_6_1x1_increase(conv4_6_3x3)
        
        conv4_6 = F.relu(conv4_5 + conv4_6_1x1_increase)
        
        # increase projection
        conv5_1_1x1_proj = self.conv5_1_1x1_proj(conv4_6)
        
        conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(conv4_6)
        conv5_1_3x3 = self.conv5_1_3x3(conv5_1_1x1_reduce)
        conv5_1_1x1_increase = self.conv5_1_1x1_increase(conv5_1_3x3)
        
        conv5_1 = F.relu(conv5_1_1x1_proj + conv5_1_1x1_increase)
<<<<<<< HEAD
        conv5_2 = self.conv5_2_bottleneck(conv5_1)
        conv5_3 = self.conv5_3_bottleneck(conv5_2)
=======
        conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(conv5_1)
        conv5_2_3x3 = self.conv5_2_3x3(conv5_2_1x1_reduce)
        conv5_2_1x1_increase = self.conv5_2_1x1_increase(conv5_2_3x3)
        
        conv5_2 = F.relu(conv5_1 + conv5_2_1x1_increase)
        conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(conv5_2)
        conv5_3_3x3 = self.conv5_3_3x3(conv5_3_1x1_reduce)
        conv5_3_1x1_increase = self.conv5_3_1x1_increase(conv5_3_3x3)
        
        conv5_3 = F.relu(conv5_2 + conv5_3_1x1_increase)
>>>>>>> master
        
        # Do global average pooling here (pyramid pooling module)
        #print("conv5_3 shape: ", x_sub4.shape)
        l, w, h = conv5_3.shape[2:]
        l = int(l)
        w = int(w)
        h = int(h)
        pool1 = F.avg_pool3d(conv5_3, (l, w, h), stride=(l, w, h))
        pool1 = F.interpolate(pool1, size=[l, w, h], mode='trilinear', align_corners=True) # resize pool1
        
        pool2 = F.avg_pool3d(conv5_3, (int(l/2), int(w/2), int(h/2)), stride=(int(l/2), int(w/2), int(l/2)))
        pool2 = F.interpolate(pool2, size=[l, w, h], mode='trilinear', align_corners=True) # resize pool2
        
        pool3 = F.avg_pool3d(conv5_3, (int(l/3), int(w/3), int(h/3)), stride=(int(l/3), int(w/3), int(h/3)))
        pool3 = F.interpolate(pool3, size=(l, w, h), mode='trilinear', align_corners=True) # resize pool3
        
        pool6 = F.avg_pool3d(conv5_3, (int(l/4), int(w/4), int(h/4)), stride=(int(l/4), int(w/4), int(h/4)))
        pool6 = F.interpolate(pool6, size=(l, w, h), mode='trilinear', align_corners=True)
        
        conv5_3_sum = conv5_3 + pool1 + pool2 + pool3 + pool6#add pooling layers and x_sub5
        
        #print("conv5_3_sum shape:", conv5_3_sum.shape)
        
        conv5_4_k1 = self.conv5_4_k1(conv5_3_sum)  # apply Batch Normalization to addition layer
        
        # perform interpolation by scale factor 2 on output
        conv5_4_interp = F.interpolate(conv5_4_k1, scale_factor=2, mode='trilinear', align_corners=True) 
        conv_sub4 = self.conv_sub4(conv5_4_interp)
        
        # perform convolution on third layer sum
        conv3_1_sub2_proj = self.conv3_1_sub2_proj(conv3_1)
        
        sub24_sum = conv3_1_sub2_proj + conv_sub4 # add outputs from two layers
        sub24_sum_interp = F.interpolate(sub24_sum, scale_factor=2, mode='trilinear', align_corners=True)
        conv_sub2 = self.conv_sub2(sub24_sum_interp)
        
        # image resolution: 1

        conv1_sub1 = self.conv1_sub1(x)
        conv2_sub1 = self.conv2_sub1(conv1_sub1)
        conv3_sub1 = self.conv3_sub1(conv2_sub1)
        conv3_sub1_proj = self.conv3_sub1_proj(conv3_sub1)

        #sub12_sum = F.relu(conv_sub2 + conv3_sub1_proj, inplace=True)
        sub12_sum = conv_sub2 + conv3_sub1_proj
        
        sub12_sum_interp = F.interpolate(sub12_sum, scale_factor=2, mode='trilinear', align_corners=True)
        
        #print("aux_1 shape:", aux_1.shape)
        #print("aux_2 shape:", aux_2.shape)
        
        # 1 image resolution
        conv6_cls = self.classification(sub12_sum_interp) # apply classification convolution
        #print("1:", conv6_cls)
        conv6_cls = self.sigmoid(conv6_cls) # apply softmax layer
   
        if self.training:
            # 1/8 image resolution
            sub4_out = self.aux_1_conv(conv5_4_interp)
            #print("aux_1:", sub4_out)
            sub4_out = self.aux_1_sigmoid(sub4_out) # apply softmax
            
            # 1/4 image resolution
            sub24_out = self.aux_2_conv(sub24_sum_interp)
            #print("aux_2:", sub24_out)
            sub24_out = self.aux_2_sigmoid(sub24_out)
            
            return (conv6_cls, sub24_out, sub4_out)
        else:
            return (conv6_cls)
<<<<<<< HEAD
     
"""

# Import all necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import numpy as np

# Important classes: BatchNormRelu3D, BatchNorm3D
from data_utils import *
=======
>>>>>>> master

# TODO: Increase maximum output size to 128 cubed, rather than the current 64 cubed tensor
# change upsample from 2 to 4 and delete some layers
# ICNet - 3D Implementation: Image-Cascade Network
<<<<<<< HEAD
class OriginalICNet(nn.Module):
    def __init__(self, num_classes):
        super(OriginalICNet, self).__init__()
=======
class ModifiedICNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedICNet, self).__init__()
>>>>>>> master
        # Initialize instance variables
        self.num_classes = num_classes
        
        #Conv3d function: 
        
        #torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,   padding_mode='zeros')
        
       # - More Notes on the input variables:
       #      - stride controls the stride for the cross-correlation.
       #      - padding controls the amount of implicit zero-paddings on both sides for padding number of points for each  dimension.
       #      - dilation controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this link has a nice visualization of what dilation does.
       #     - groups controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups. 
       #         - At groups=1, all inputs are convolved to all outputs.
       #         - At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels, and producing half the output channels, and both subsequently concatenated. 
       #         - At groups= in_channels, each input channel is convolved with its own set of filters, of size [out_channels/in_channels]
        
        
        # BatchNormRelu3D and BatchNorm3D(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=False)
        
        # Convolutions for 1/2 Resolution of Image
        
        # Do trilinear interpolation to get 1/2 resolution
        # s2 = stride 2
        self.conv1_1_3x3_s2 = BatchNormRelu3D(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv1_2_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv1_3_3x3 = BatchNormRelu3D(8, 16,  3, 1, padding=1)
        
        # max pooling does not affect the number of out channels, but rather reduces the spatial dimensions of the output for simplicity in the model
        self.pool1_3x3_s2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Apply these convolutions to the max pooling layer
        self.conv2_1_1x1_proj = BatchNorm3D(16, 32, 1, 1)
        
        #self.conv2_1_bottleneck = BottleNeck(128, 32, 1) 
        # bottleneck
        self.conv2_1_1x1_reduce = BatchNormRelu3D(16, 8, 1, 1)
        self.conv2_1_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv2_1_1x1_increase = BatchNorm3D(8, 32, 1, 1)
        
        # conv2_2
        #self.conv2_2_bottleneck = BottleNeck(32, 8, dilation=1) # bottleneck
        self.conv2_2_1x1_reduce = BatchNormRelu3D(32, 8, 1, 1)
        self.conv2_2_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv2_2_1x1_increase = BatchNorm3D(8, 32, 1, 1)
        
        # conv2_3
        #self.conv2_3_bottleneck = BottleNeck(32, 8, dilation=1)
        self.conv2_3_1x1_reduce = BatchNormRelu3D(32, 8, 1, 1)
        self.conv2_3_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv2_3_1x1_increase = BatchNorm3D(8, 32, 1, 1)
        
        # conv3_1
        self.conv3_1_1x1_proj = BatchNorm3D(32, 64, 1, stride=2) # stride 2
        
        # bottleneck building block
        self.conv3_1_1x1_reduce = BatchNormRelu3D(32, 16, 1, 2) # stride 2
        self.conv3_1_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_1_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # self.conv3_1_bottleneck = BottleNeckConv3_1(256, 64, 1)
        
        # Convolutions for 1/4 Image Resolution
        
        # conv3_2 - bottleneck building block
        #self.conv3_2_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_2_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv3_2_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_2_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv3_3 - bottleneck building block
        #self.conv3_3_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_3_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv3_3_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_3_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv3_4
        #self.conv3_4_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_4_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv3_4_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_4_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv4_1
        self.conv4_1_1x1_proj = BatchNorm3D(64, 128, 1, 1)
        
        #self.conv4_1_bottleneck = BottleNeck(512, 128, dilation=2)
        self.conv4_1_1x1_reduce = BatchNormRelu3D(64, 32, 1, 1)
        self.conv4_1_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_1_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_2
        #self.conv4_2_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_2_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_2_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_2_1x1_increase = BatchNorm3D(32, 128, 1, 1)
<<<<<<< HEAD
        
        # conv4_3
        #self.conv4_3_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_3_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_3_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_3_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_4
        #self.conv4_4_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_4_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_4_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_4_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_5
        #self.conv4_5_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_5_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_5_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_5_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_6
        #self.conv4_6_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_6_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_6_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_6_1x1_increase = BatchNorm3D(32, 128, 1, 1)
    
        # conv5_1
        self.conv5_1_1x1_proj = BatchNorm3D(128, 256, 1, 1)
        
        # bottleneck conv5-1
        self.conv5_1_1x1_reduce = BatchNormRelu3D(128, 64, 1, 1)
        self.conv5_1_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_1_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv5_2
        #self.conv5_2_bottleneck = BottleNeck(256, 64, dilation=4)
        self.conv5_2_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv5_2_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_2_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv5_3
        #self.conv5_3_bottleneck = BottleNeck(256, 64, dilation=4)
        self.conv5_3_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv5_3_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_3_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # Global average pooling here
        
        # conv5_3_sum
        self.conv5_4_k1 = BatchNorm3D(256, 64, 1, 1)
        self.conv_sub4 = BatchNorm3D(64, 32, 3, 1, padding=2, dilation=2) # dilation rate = 2
        self.conv3_1_sub2_proj = BatchNorm3D(64, 32, 1, 1)
        self.conv_sub2 = BatchNorm3D(32, 32, 3, 1, padding=2, dilation=2) # dilation rate = 2
        
        # Convolutions for 1 image resolution
        self.conv1_sub1 = BatchNormRelu3D(1, 8, 3, stride=2, padding=1) # stride = 2
        self.conv2_sub1 = BatchNormRelu3D(8, 8, 3, stride=2, padding=1) # stride = 2
        self.conv3_sub1 = BatchNormRelu3D(8, 16, 3, stride=2, padding=1) # stride = 2
        self.conv3_sub1_proj = BatchNorm3D(16, 32, 1, 1)
        
        self.aux_1_conv = nn.Conv3d(64, self.num_classes, 1, 1)
        self.aux_2_conv = nn.Conv3d(32, self.num_classes, 1, 1)
        self.classification = nn.Conv3d(32, self.num_classes, 1, 1)
        
        self.sigmoid = nn.Softmax(dim=1)
        self.aux_1_sigmoid = nn.Softmax(dim=1)
        self.aux_2_sigmoid = nn.Softmax(dim=1)
        
        # define loss function (self.loss = ...)
        
    # Input x: (shape = (frames, height, width))
    def forward(self, x):
        #print(x.shape)
        l, h, w = x.shape[2:] 

        # image resolution: 1/2 - L, H, W -> L/2, H/2, W/2
        data_sub2 = F.interpolate(x, scale_factor = 1/2, mode="trilinear", align_corners=True)

        # L/2, H/2, W/2 -> L/4, H/4, W/4
        conv1_1_3x3_s2 = self.conv1_1_3x3_s2(data_sub2)
        conv1_2_3x3 = self.conv1_2_3x3(conv1_1_3x3_s2)
        conv1_3_3x3 = self.conv1_3_3x3(conv1_2_3x3)

        # L/4, H/4, W/4 -> L/8, H/8, W/8
        pool1_3x3_s2 = self.pool1_3x3_s2(conv1_3_3x3)

        # L/8, H/8, W/8 -> L/16, H/16, W/16
        conv2_1_1x1_proj = self.conv2_1_1x1_proj(pool1_3x3_s2)
        
        # bottleneck
        conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_3x3 = self.conv2_1_3x3(conv2_1_1x1_reduce)
        conv2_1_1x1_increase = self.conv2_1_1x1_increase(conv2_1_3x3)
        
        # Add residual of conv2_1_1x1 projection to bottleneck of conv2_1
        conv2_1 = F.relu(conv2_1_1x1_proj + conv2_1_1x1_increase)
        conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(conv2_1)
        conv2_2_3x3 = self.conv2_2_3x3(conv2_2_1x1_reduce)
        conv2_2_1x1_increase = self.conv2_2_1x1_increase(conv2_2_3x3)
        
        conv2_2 = F.relu(conv2_1 + conv2_2_1x1_increase)
        conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(conv2_2)
        conv2_3_3x3 = self.conv2_3_3x3(conv2_3_1x1_reduce)
        conv2_3_1x1_increase = self.conv2_3_1x1_increase(conv2_3_3x3)
        
        conv2_3 = F.relu(conv2_2 + conv2_3_1x1_increase)
        
        conv3_1_1x1_proj = self.conv3_1_1x1_proj(conv2_3)
        
        # bottleneck residual block on conv2_3
        conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(conv2_3)
        conv3_1_3x3 = self.conv3_1_3x3(conv3_1_1x1_reduce)
        conv3_1_1x1_increase = self.conv3_1_1x1_increase(conv3_1_3x3)
        
        #conv3_1 = conv3_1_1x1_proj + conv3_1_1x1_increase
        conv3_1 = F.relu(conv3_1_1x1_proj + conv3_1_1x1_increase)
        
        # image resolution: 1/4 - L/16, H/16, W/16 -> L/32, H/32, W/32
        conv3_1_sub4 = F.interpolate(conv3_1, scale_factor=1/2, mode="trilinear", align_corners=True)
        
        #print("conv3_1 - x_sub4 shape:", x_sub4.shape)
        
        conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(conv3_1_sub4)
        conv3_2_3x3 = self.conv3_2_3x3(conv3_2_1x1_reduce)
        conv3_2_1x1_increase = self.conv3_2_1x1_increase(conv3_2_3x3)
        
        conv3_2 = F.relu(conv3_1_sub4 + conv3_2_1x1_increase)
        conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(conv3_2)
        conv3_3_3x3 = self.conv3_3_3x3(conv3_3_1x1_reduce)
        conv3_3_1x1_increase = self.conv3_3_1x1_increase(conv3_3_3x3)
        
        conv3_3 = F.relu(conv3_2 + conv3_3_1x1_increase)
        conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(conv3_3)
        conv3_4_3x3 = self.conv3_4_3x3(conv3_4_1x1_reduce)
        conv3_4_1x1_increase = self.conv3_4_1x1_increase(conv3_4_3x3)
        
        conv3_4 = F.relu(conv3_3 + conv3_4_1x1_increase)
        
        # increase projection
        conv4_1_1x1_proj = self.conv4_1_1x1_proj(conv3_4)
        
        # bottlenecks
        conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(conv3_4)
        conv4_1_3x3 = self.conv4_1_3x3(conv4_1_1x1_reduce)
        conv4_1_1x1_increase = self.conv4_1_1x1_increase(conv4_1_3x3)
        
        conv4_1 = F.relu(conv4_1_1x1_proj + conv4_1_1x1_increase)
        #conv4_1 = conv4_1_1x1_proj + conv4_1_1x1_increase
        conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(conv4_1)
        conv4_2_3x3 = self.conv4_2_3x3(conv4_2_1x1_reduce)
        conv4_2_1x1_increase = self.conv4_2_1x1_increase(conv4_2_3x3)
        
        conv4_2 = F.relu(conv4_1 + conv4_2_1x1_increase)
        conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(conv4_2)
        conv4_3_3x3 = self.conv4_3_3x3(conv4_3_1x1_reduce)
        conv4_3_1x1_increase = self.conv4_3_1x1_increase(conv4_3_3x3)
        
        conv4_3 = F.relu(conv4_2 + conv4_3_1x1_increase)
        conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(conv4_3)
        conv4_4_3x3 = self.conv4_4_3x3(conv4_4_1x1_reduce)
        conv4_4_1x1_increase = self.conv4_4_1x1_increase(conv4_4_3x3)
        
        conv4_4 = F.relu(conv4_3 + conv4_4_1x1_increase)
        conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(conv4_4)
        conv4_5_3x3 = self.conv4_5_3x3(conv4_5_1x1_reduce)
        conv4_5_1x1_increase = self.conv4_5_1x1_increase(conv4_5_3x3)
        
        conv4_5 = F.relu(conv4_4 + conv4_5_1x1_increase)
        conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(conv4_5)
        conv4_6_3x3 = self.conv4_6_3x3(conv4_6_1x1_reduce)
        conv4_6_1x1_increase = self.conv4_6_1x1_increase(conv4_6_3x3)
        
        conv4_6 = F.relu(conv4_5 + conv4_6_1x1_increase)
        
        # increase projection
        conv5_1_1x1_proj = self.conv5_1_1x1_proj(conv4_6)
        
        conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(conv4_6)
        conv5_1_3x3 = self.conv5_1_3x3(conv5_1_1x1_reduce)
        conv5_1_1x1_increase = self.conv5_1_1x1_increase(conv5_1_3x3)
        
        conv5_1 = F.relu(conv5_1_1x1_proj + conv5_1_1x1_increase)
        conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(conv5_1)
        conv5_2_3x3 = self.conv5_2_3x3(conv5_2_1x1_reduce)
        conv5_2_1x1_increase = self.conv5_2_1x1_increase(conv5_2_3x3)
        
        conv5_2 = F.relu(conv5_1 + conv5_2_1x1_increase)
        conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(conv5_2)
        conv5_3_3x3 = self.conv5_3_3x3(conv5_3_1x1_reduce)
        conv5_3_1x1_increase = self.conv5_3_1x1_increase(conv5_3_3x3)
        
        conv5_3 = F.relu(conv5_2 + conv5_3_1x1_increase)
        
        # Do global average pooling here (pyramid pooling module)
        #print("conv5_3 shape: ", x_sub4.shape)
        l, w, h = conv5_3.shape[2:]
        l = int(l)
        w = int(w)
        h = int(h)
        pool1 = F.avg_pool3d(conv5_3, (l, w, h), stride=(l, w, h))
        pool1 = F.interpolate(pool1, size=[l, w, h], mode='trilinear', align_corners=True) # resize pool1
        
        pool2 = F.avg_pool3d(conv5_3, (int(l/2), int(w/2), int(h/2)), stride=(int(l/2), int(w/2), int(l/2)))
        pool2 = F.interpolate(pool2, size=[l, w, h], mode='trilinear', align_corners=True) # resize pool2
        
        pool3 = F.avg_pool3d(conv5_3, (int(l/3), int(w/3), int(h/3)), stride=(int(l/3), int(w/3), int(h/3)))
        pool3 = F.interpolate(pool3, size=(l, w, h), mode='trilinear', align_corners=True) # resize pool3
        
        pool6 = F.avg_pool3d(conv5_3, (int(l/4), int(w/4), int(h/4)), stride=(int(l/4), int(w/4), int(h/4)))
        pool6 = F.interpolate(pool6, size=(l, w, h), mode='trilinear', align_corners=True)
        
        conv5_3_sum = conv5_3 + pool1 + pool2 + pool3 + pool6#add pooling layers and x_sub5
        
        #print("conv5_3_sum shape:", conv5_3_sum.shape)
        
        conv5_4_k1 = self.conv5_4_k1(conv5_3_sum)  # apply Batch Normalization to addition layer
        
        # perform interpolation by scale factor 2 on output
        conv5_4_interp = F.interpolate(conv5_4_k1, scale_factor=2, mode='trilinear', align_corners=True) 
        conv_sub4 = self.conv_sub4(conv5_4_interp)
        
        # perform convolution on third layer sum
        conv3_1_sub2_proj = self.conv3_1_sub2_proj(conv3_1)
        
        sub24_sum = conv3_1_sub2_proj + conv_sub4 # add outputs from two layers
        sub24_sum_interp = F.interpolate(sub24_sum, scale_factor=2, mode='trilinear', align_corners=True)
        conv_sub2 = self.conv_sub2(sub24_sum_interp)
        
        # image resolution: 1

        conv1_sub1 = self.conv1_sub1(x)
        conv2_sub1 = self.conv2_sub1(conv1_sub1)
        conv3_sub1 = self.conv3_sub1(conv2_sub1)
        conv3_sub1_proj = self.conv3_sub1_proj(conv3_sub1)

        #sub12_sum = F.relu(conv_sub2 + conv3_sub1_proj, inplace=True)
        sub12_sum = conv_sub2 + conv3_sub1_proj
        
        sub12_sum_interp = F.interpolate(sub12_sum, scale_factor=2, mode='trilinear', align_corners=True)
        
        #print("aux_1 shape:", aux_1.shape)
        #print("aux_2 shape:", aux_2.shape)
        
        # 1 image resolution
        conv6_cls = self.classification(sub12_sum_interp) # apply classification convolution
=======
        
        # conv4_3
        #self.conv4_3_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_3_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_3_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_3_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_4
        #self.conv4_4_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_4_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_4_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_4_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_5
        #self.conv4_5_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_5_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_5_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_5_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_6
        #self.conv4_6_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_6_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_6_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_6_1x1_increase = BatchNorm3D(32, 128, 1, 1)
    
        # conv5_1
        self.conv5_1_1x1_proj = BatchNorm3D(128, 256, 1, 1)
        
        # bottleneck conv5-1
        self.conv5_1_1x1_reduce = BatchNormRelu3D(128, 64, 1, 1)
        self.conv5_1_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_1_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv5_2
        #self.conv5_2_bottleneck = BottleNeck(256, 64, dilation=4)
        self.conv5_2_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv5_2_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_2_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv5_3
        #self.conv5_3_bottleneck = BottleNeck(256, 64, dilation=4)
        self.conv5_3_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv5_3_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_3_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # Global average pooling here
        
        # conv5_3_sum
        self.conv5_4_k1 = BatchNorm3D(256, 64, 1, 1)
        self.conv_sub4 = BatchNorm3D(64, 32, 3, stride=2, padding=2, dilation=2) # dilation rate = 2
        self.conv3_1_sub2_proj = BatchNorm3D(64, 32, 1, stride=1)
        self.conv_sub2 = BatchNorm3D(32, 32, 3, stride=2, padding=2, dilation=2) # dilation rate = 2
        
        # Convolutions for 1 image resolution
        self.conv1_sub1 = BatchNormRelu3D(1, 8, 3, stride=2, padding=1) # stride = 2
        self.conv2_sub1 = BatchNormRelu3D(8, 8, 3, stride=2, padding=1) # stride = 2
        self.conv3_sub1 = BatchNormRelu3D(8, 16, 3, stride=2, padding=1) # stride = 2
        self.conv3_sub1_proj = BatchNorm3D(16, 32, 1, 1)
        
        self.sub12_sum_transpose = nn.ConvTranspose3d(32, 32, kernel_size=4, stride=4)
        
        self.aux_1_conv = nn.Conv3d(64, self.num_classes, 1, 1)
        self.aux_2_conv = nn.Conv3d(32, self.num_classes, 1, 1)
        self.classification = nn.Conv3d(32, self.num_classes, 1, 1)
        
        self.sigmoid = nn.Softmax(dim=1)
        self.aux_1_sigmoid = nn.Softmax(dim=1)
        self.aux_2_sigmoid = nn.Softmax(dim=1)
        
        # define loss function (self.loss = ...)
        
    # Input x: (shape = (frames, height, width))
    def forward(self, x):
        #print(x.shape)
        #l, h, w = x.shape[2:] 

        # image resolution: 1/2 - L, H, W -> L/2, H/2, W/2
        data_sub2 = F.interpolate(x, scale_factor = 1/2, mode="trilinear", align_corners=True)

        # L/2, H/2, W/2 -> L/4, H/4, W/4
        conv1_1_3x3_s2 = self.conv1_1_3x3_s2(data_sub2)
        conv1_2_3x3 = self.conv1_2_3x3(conv1_1_3x3_s2)
        conv1_3_3x3 = self.conv1_3_3x3(conv1_2_3x3)

        # L/4, H/4, W/4 -> L/8, H/8, W/8
        pool1_3x3_s2 = self.pool1_3x3_s2(conv1_3_3x3)

        # L/8, H/8, W/8 -> L/16, H/16, W/16
        conv2_1_1x1_proj = self.conv2_1_1x1_proj(pool1_3x3_s2)
        
        # bottleneck
        conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_3x3 = self.conv2_1_3x3(conv2_1_1x1_reduce)
        conv2_1_1x1_increase = self.conv2_1_1x1_increase(conv2_1_3x3)
        
        # Add residual of conv2_1_1x1 projection to bottleneck of conv2_1
        conv2_1 = F.relu(conv2_1_1x1_proj + conv2_1_1x1_increase)
        conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(conv2_1)
        conv2_2_3x3 = self.conv2_2_3x3(conv2_2_1x1_reduce)
        conv2_2_1x1_increase = self.conv2_2_1x1_increase(conv2_2_3x3)
        
        conv2_2 = F.relu(conv2_1 + conv2_2_1x1_increase)
        conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(conv2_2)
        conv2_3_3x3 = self.conv2_3_3x3(conv2_3_1x1_reduce)
        conv2_3_1x1_increase = self.conv2_3_1x1_increase(conv2_3_3x3)
        
        conv2_3 = F.relu(conv2_2 + conv2_3_1x1_increase)
        
        conv3_1_1x1_proj = self.conv3_1_1x1_proj(conv2_3)
        
        # bottleneck residual block on conv2_3
        conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(conv2_3)
        conv3_1_3x3 = self.conv3_1_3x3(conv3_1_1x1_reduce)
        conv3_1_1x1_increase = self.conv3_1_1x1_increase(conv3_1_3x3)
        
        #conv3_1 = conv3_1_1x1_proj + conv3_1_1x1_increase
        conv3_1 = F.relu(conv3_1_1x1_proj + conv3_1_1x1_increase)
        
        # image resolution: 1/4 - L/16, H/16, W/16 -> L/32, H/32, W/32
        conv3_1_sub4 = F.interpolate(conv3_1, scale_factor=1/2, mode="trilinear", align_corners=True)
        
        #print("conv3_1 - x_sub4 shape:", x_sub4.shape)
        
        conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(conv3_1_sub4)
        conv3_2_3x3 = self.conv3_2_3x3(conv3_2_1x1_reduce)
        conv3_2_1x1_increase = self.conv3_2_1x1_increase(conv3_2_3x3)
        
        conv3_2 = F.relu(conv3_1_sub4 + conv3_2_1x1_increase)
        conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(conv3_2)
        conv3_3_3x3 = self.conv3_3_3x3(conv3_3_1x1_reduce)
        conv3_3_1x1_increase = self.conv3_3_1x1_increase(conv3_3_3x3)
        
        conv3_3 = F.relu(conv3_2 + conv3_3_1x1_increase)
        conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(conv3_3)
        conv3_4_3x3 = self.conv3_4_3x3(conv3_4_1x1_reduce)
        conv3_4_1x1_increase = self.conv3_4_1x1_increase(conv3_4_3x3)
        
        conv3_4 = F.relu(conv3_3 + conv3_4_1x1_increase)
        
        # increase projection
        conv4_1_1x1_proj = self.conv4_1_1x1_proj(conv3_4)
        
        # bottlenecks
        conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(conv3_4)
        conv4_1_3x3 = self.conv4_1_3x3(conv4_1_1x1_reduce)
        conv4_1_1x1_increase = self.conv4_1_1x1_increase(conv4_1_3x3)
        
        conv4_1 = F.relu(conv4_1_1x1_proj + conv4_1_1x1_increase)
        #conv4_1 = conv4_1_1x1_proj + conv4_1_1x1_increase
        conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(conv4_1)
        conv4_2_3x3 = self.conv4_2_3x3(conv4_2_1x1_reduce)
        conv4_2_1x1_increase = self.conv4_2_1x1_increase(conv4_2_3x3)
        
        conv4_2 = F.relu(conv4_1 + conv4_2_1x1_increase)
        conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(conv4_2)
        conv4_3_3x3 = self.conv4_3_3x3(conv4_3_1x1_reduce)
        conv4_3_1x1_increase = self.conv4_3_1x1_increase(conv4_3_3x3)
        
        conv4_3 = F.relu(conv4_2 + conv4_3_1x1_increase)
        conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(conv4_3)
        conv4_4_3x3 = self.conv4_4_3x3(conv4_4_1x1_reduce)
        conv4_4_1x1_increase = self.conv4_4_1x1_increase(conv4_4_3x3)
        
        conv4_4 = F.relu(conv4_3 + conv4_4_1x1_increase)
        conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(conv4_4)
        conv4_5_3x3 = self.conv4_5_3x3(conv4_5_1x1_reduce)
        conv4_5_1x1_increase = self.conv4_5_1x1_increase(conv4_5_3x3)
        
        conv4_5 = F.relu(conv4_4 + conv4_5_1x1_increase)
        conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(conv4_5)
        conv4_6_3x3 = self.conv4_6_3x3(conv4_6_1x1_reduce)
        conv4_6_1x1_increase = self.conv4_6_1x1_increase(conv4_6_3x3)
        
        conv4_6 = F.relu(conv4_5 + conv4_6_1x1_increase)
        
        # increase projection
        conv5_1_1x1_proj = self.conv5_1_1x1_proj(conv4_6)
        
        conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(conv4_6)
        conv5_1_3x3 = self.conv5_1_3x3(conv5_1_1x1_reduce)
        conv5_1_1x1_increase = self.conv5_1_1x1_increase(conv5_1_3x3)
        
        conv5_1 = F.relu(conv5_1_1x1_proj + conv5_1_1x1_increase)
        conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(conv5_1)
        conv5_2_3x3 = self.conv5_2_3x3(conv5_2_1x1_reduce)
        conv5_2_1x1_increase = self.conv5_2_1x1_increase(conv5_2_3x3)
        
        conv5_2 = F.relu(conv5_1 + conv5_2_1x1_increase)
        conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(conv5_2)
        conv5_3_3x3 = self.conv5_3_3x3(conv5_3_1x1_reduce)
        conv5_3_1x1_increase = self.conv5_3_1x1_increase(conv5_3_3x3)
        
        conv5_3 = F.relu(conv5_2 + conv5_3_1x1_increase)
        
        # Do global average pooling here (pyramid pooling module)
        #print("conv5_3 shape: ", x_sub4.shape)
        l, w, h = conv5_3.shape[2:]
        l = int(l)
        w = int(w)
        h = int(h)
        pool1 = F.avg_pool3d(conv5_3, (l, w, h), stride=(l, w, h))
        pool1 = F.interpolate(pool1, size=[l, w, h], mode='trilinear', align_corners=True) # resize pool1
        
        pool2 = F.avg_pool3d(conv5_3, (int(l/2), int(w/2), int(h/2)), stride=(int(l/2), int(w/2), int(l/2)))
        pool2 = F.interpolate(pool2, size=[l, w, h], mode='trilinear', align_corners=True) # resize pool2
        
        pool3 = F.avg_pool3d(conv5_3, (int(l/3), int(w/3), int(h/3)), stride=(int(l/3), int(w/3), int(h/3)))
        pool3 = F.interpolate(pool3, size=(l, w, h), mode='trilinear', align_corners=True) # resize pool3
        
        pool6 = F.avg_pool3d(conv5_3, (int(l/4), int(w/4), int(h/4)), stride=(int(l/4), int(w/4), int(h/4)))
        pool6 = F.interpolate(pool6, size=(l, w, h), mode='trilinear', align_corners=True)
        
        conv5_3_sum = conv5_3 + pool1 + pool2 + pool3 + pool6#add pooling layers and x_sub5
        
        #print("conv5_3_sum shape:", conv5_3_sum.shape)
        
        conv5_4_k1 = self.conv5_4_k1(conv5_3_sum)  # apply Batch Normalization to addition layer
        
        # perform interpolation by scale factor 2 on output
        conv5_4_interp = F.interpolate(conv5_4_k1, scale_factor=4, mode='trilinear', align_corners=True) 
        conv_sub4 = self.conv_sub4(conv5_4_interp)
        
        # perform convolution on third layer sum
        #conv3_1 = F.interpolate(conv3_1, scale_factor=2, mode='trilinear', align_corners=True)
        conv3_1_sub2_proj = self.conv3_1_sub2_proj(conv3_1)
        #print("conv3_1_sub2_proj:", conv3_1_sub2_proj.shape)
        #print("conv_sub4:", conv_sub4.shape)
        sub24_sum = conv3_1_sub2_proj + conv_sub4 # add outputs from two layers
        sub24_sum_interp = F.interpolate(sub24_sum, scale_factor=4, mode='trilinear', align_corners=True)
        conv_sub2 = self.conv_sub2(sub24_sum_interp)
       
        # image resolution: 1
        #print("x:", x.shape)
        conv1_sub1 = self.conv1_sub1(x)
        #print("conv1_sub1:", conv1_sub1.shape)
        conv2_sub1 = self.conv2_sub1(conv1_sub1)
        #print("conv2_sub1:", conv2_sub1.shape)
        conv3_sub1 = self.conv3_sub1(conv2_sub1)
        #print("conv3_sub1:", conv3_sub1.shape)
        conv3_sub1_proj = self.conv3_sub1_proj(conv3_sub1)
        #print("conv_sub2:", conv_sub2.shape)
        #print("conv3_sub1_proj:", conv3_sub1_proj.shape)
        #sub12_sum = F.relu(conv_sub2 + conv3_sub1_proj, inplace=True)
       
        sub12_sum = conv_sub2 + conv3_sub1_proj
        #print(sub12_sum.shape)
        sub12_sum_transpose = self.sub12_sum_transpose(sub12_sum)
        #print(sub12_sum_transpose.shape)
        #sub12_sum_interp = F.interpolate(sub12_sum, scale_factor=4, mode='trilinear', align_corners=True)
        
        # 1 image resolution
        conv6_cls = self.classification(sub12_sum_transpose) # apply classification convolution
>>>>>>> master
        #print("1:", conv6_cls)
        conv6_cls = self.sigmoid(conv6_cls) # apply softmax layer
   
        if self.training:
            # 1/8 image resolution
            sub4_out = self.aux_1_conv(conv5_4_interp)
            #print("aux_1:", sub4_out)
            sub4_out = self.aux_1_sigmoid(sub4_out) # apply softmax
            
            # 1/4 image resolution
            sub24_out = self.aux_2_conv(sub24_sum_interp)
            #print("aux_2:", sub24_out)
            sub24_out = self.aux_2_sigmoid(sub24_out)
            
            return (conv6_cls, sub24_out, sub4_out)
        else:
            return (conv6_cls)
<<<<<<< HEAD

# TODO: Increase maximum output size to 128 cubed, rather than the current 64 cubed tensor
# change upsample from 2 to 4 and delete some layers
# ICNet - 3D Implementation: Image-Cascade Network
class ModifiedICNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedICNet, self).__init__()
=======
        
class FullResolutionICNet_MoreFeatures(nn.Module):
    def __init__(self, num_classes):
        super(FullResolutionICNet_MoreFeatures, self).__init__()
>>>>>>> master
        # Initialize instance variables
        self.num_classes = num_classes
        
        #Conv3d function: 
<<<<<<< HEAD
=======
        # x = conv3d (x, 1, 8) # 256
        # x = conv3d (x, 8, 8) # 256
        # x = conv3d (x, 8, 8) # 256
        # x = conv3d (x, 8, 16, stride = 2) or what ever downsample, pooling # 256 ->128

        # 2_featmap_up = upsample (2_featmap, scale = 4) # 32->128
        # 2_featmap_adj = conv3d (2_featmap_up, num of filter size last model, 16) # 128
        # x += 2_featmap_adj # 128
        # x = conv3d (x, 16, 16) # 128

        # x = ConvTranspose3d (x, 16, 8, stride = 2) # 128->256
        # x = conv3d (x, 8, 3) #256 * 3 classes
        # x = Softmax (x, dim=1)
>>>>>>> master
        
        #torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,   padding_mode='zeros')
        
       # - More Notes on the input variables:
       #      - stride controls the stride for the cross-correlation.
       #      - padding controls the amount of implicit zero-paddings on both sides for padding number of points for each  dimension.
       #      - dilation controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this link has a nice visualization of what dilation does.
       #     - groups controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups. 
       #         - At groups=1, all inputs are convolved to all outputs.
       #         - At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels, and producing half the output channels, and both subsequently concatenated. 
       #         - At groups= in_channels, each input channel is convolved with its own set of filters, of size [out_channels/in_channels]
        
        
        # BatchNormRelu3D and BatchNorm3D(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=False)
        
        # Convolutions for 1/2 Resolution of Image
<<<<<<< HEAD
        
        # Do trilinear interpolation to get 1/2 resolution
        # s2 = stride 2
        self.conv1_1_3x3_s2 = BatchNormRelu3D(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv1_2_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv1_3_3x3 = BatchNormRelu3D(8, 16,  3, 1, padding=1)
        
        # max pooling does not affect the number of out channels, but rather reduces the spatial dimensions of the output for simplicity in the model
        self.pool1_3x3_s2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Apply these convolutions to the max pooling layer
        self.conv2_1_1x1_proj = BatchNorm3D(16, 32, 1, 1)
        
        #self.conv2_1_bottleneck = BottleNeck(128, 32, 1) 
        # bottleneck
        self.conv2_1_1x1_reduce = BatchNormRelu3D(16, 8, 1, 1)
        self.conv2_1_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv2_1_1x1_increase = BatchNorm3D(8, 32, 1, 1)
        
        # conv2_2
        #self.conv2_2_bottleneck = BottleNeck(32, 8, dilation=1) # bottleneck
        self.conv2_2_1x1_reduce = BatchNormRelu3D(32, 8, 1, 1)
        self.conv2_2_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv2_2_1x1_increase = BatchNorm3D(8, 32, 1, 1)
        
        # conv2_3
        #self.conv2_3_bottleneck = BottleNeck(32, 8, dilation=1)
        self.conv2_3_1x1_reduce = BatchNormRelu3D(32, 8, 1, 1)
        self.conv2_3_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv2_3_1x1_increase = BatchNorm3D(8, 32, 1, 1)
        
        # conv3_1
        self.conv3_1_1x1_proj = BatchNorm3D(32, 64, 1, stride=2) # stride 2
        
        # bottleneck building block
        self.conv3_1_1x1_reduce = BatchNormRelu3D(32, 16, 1, 2) # stride 2
        self.conv3_1_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_1_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # self.conv3_1_bottleneck = BottleNeckConv3_1(256, 64, 1)
        
        # Convolutions for 1/4 Image Resolution
        
        # conv3_2 - bottleneck building block
        #self.conv3_2_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_2_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv3_2_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_2_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv3_3 - bottleneck building block
        #self.conv3_3_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_3_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv3_3_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_3_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv3_4
        #self.conv3_4_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_4_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv3_4_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_4_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv4_1
        self.conv4_1_1x1_proj = BatchNorm3D(64, 128, 1, 1)
        
        #self.conv4_1_bottleneck = BottleNeck(512, 128, dilation=2)
        self.conv4_1_1x1_reduce = BatchNormRelu3D(64, 32, 1, 1)
        self.conv4_1_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_1_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_2
        #self.conv4_2_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_2_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_2_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_2_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_3
        #self.conv4_3_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_3_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_3_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_3_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_4
        #self.conv4_4_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_4_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_4_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_4_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_5
        #self.conv4_5_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_5_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_5_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_5_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_6
        #self.conv4_6_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_6_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_6_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_6_1x1_increase = BatchNorm3D(32, 128, 1, 1)
    
        # conv5_1
        self.conv5_1_1x1_proj = BatchNorm3D(128, 256, 1, 1)
        
        # bottleneck conv5-1
        self.conv5_1_1x1_reduce = BatchNormRelu3D(128, 64, 1, 1)
        self.conv5_1_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_1_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv5_2
        #self.conv5_2_bottleneck = BottleNeck(256, 64, dilation=4)
        self.conv5_2_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv5_2_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_2_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv5_3
        #self.conv5_3_bottleneck = BottleNeck(256, 64, dilation=4)
        self.conv5_3_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv5_3_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_3_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # Global average pooling here
        
        # conv5_3_sum
        self.conv5_4_k1 = BatchNorm3D(256, 64, 1, 1)
        self.conv_sub4 = BatchNorm3D(64, 32, 3, stride=2, padding=2, dilation=2) # dilation rate = 2
        self.conv3_1_sub2_proj = BatchNorm3D(64, 32, 1, stride=1)
        self.conv_sub2 = BatchNorm3D(32, 32, 3, stride=2, padding=2, dilation=2) # dilation rate = 2
        
        # Convolutions for 1 image resolution
        self.conv1_sub1 = BatchNormRelu3D(1, 8, 3, stride=2, padding=1) # stride = 2
        self.conv2_sub1 = BatchNormRelu3D(8, 8, 3, stride=2, padding=1) # stride = 2
        self.conv3_sub1 = BatchNormRelu3D(8, 16, 3, stride=2, padding=1) # stride = 2
        self.conv3_sub1_proj = BatchNorm3D(16, 32, 1, 1)
        
        self.sub12_sum_transpose = nn.ConvTranspose3d(32, 32, kernel_size=4, stride=4)
        
        self.aux_1_conv = nn.Conv3d(64, self.num_classes, 1, 1)
        self.aux_2_conv = nn.Conv3d(32, self.num_classes, 1, 1)
        self.classification = nn.Conv3d(32, self.num_classes, 1, 1)
=======
        
        # Do trilinear interpolation to get 1/2 resolution
        # s2 = stride 2
        self.conv1_1_3x3_s2 = BatchNormRelu3D(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv1_2_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv1_3_3x3 = BatchNormRelu3D(16, 32,  3, 1, padding=1)
        
        # max pooling does not affect the number of out channels, but rather reduces the spatial dimensions of the output for simplicity in the model
        self.pool1_3x3_s2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Apply these convolutions to the max pooling layer
        self.conv2_1_1x1_proj = BatchNorm3D(32, 64, 1, 1)
        
        #self.conv2_1_bottleneck = BottleNeck(128, 32, 1) 
        # bottleneck
        self.conv2_1_1x1_reduce = BatchNormRelu3D(32,16, 1, 1)
        self.conv2_1_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv2_1_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv2_2
        #self.conv2_2_bottleneck = BottleNeck(32, 8, dilation=1) # bottleneck
        self.conv2_2_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv2_2_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv2_2_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv2_3
        #self.conv2_3_bottleneck = BottleNeck(32, 8, dilation=1)
        self.conv2_3_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv2_3_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv2_3_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv3_1
        self.conv3_1_1x1_proj = BatchNorm3D(64, 128, 1, stride=2) # stride 2
        
        # bottleneck building block
        self.conv3_1_1x1_reduce = BatchNormRelu3D(64, 32, 1, 2) # stride 2
        self.conv3_1_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=1)
        self.conv3_1_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # self.conv3_1_bottleneck = BottleNeckConv3_1(256, 64, 1)
        
        # Convolutions for 1/4 Image Resolution
        
        # conv3_2 - bottleneck building block
        #self.conv3_2_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_2_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv3_2_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=1)
        self.conv3_2_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv3_3 - bottleneck building block
        #self.conv3_3_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_3_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv3_3_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=1)
        self.conv3_3_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv3_4
        #self.conv3_4_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_4_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv3_4_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=1)
        self.conv3_4_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_1
        self.conv4_1_1x1_proj = BatchNorm3D(128, 256, 1, 1)
        
        #self.conv4_1_bottleneck = BottleNeck(512, 128, dilation=2)
        self.conv4_1_1x1_reduce = BatchNormRelu3D(128, 64, 1, 1)
        self.conv4_1_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=2, dilation=2)
        self.conv4_1_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv4_2
        #self.conv4_2_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_2_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv4_2_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=2, dilation=2)
        self.conv4_2_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv4_3
        #self.conv4_3_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_3_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv4_3_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=2, dilation=2)
        self.conv4_3_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv4_4
        #self.conv4_4_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_4_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv4_4_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=2, dilation=2)
        self.conv4_4_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv4_5
        #self.conv4_5_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_5_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv4_5_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=2, dilation=2)
        self.conv4_5_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv4_6
        #self.conv4_6_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_6_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv4_6_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=2, dilation=2)
        self.conv4_6_1x1_increase = BatchNorm3D(64, 256, 1, 1)
    
        # conv5_1
        self.conv5_1_1x1_proj = BatchNorm3D(256, 512, 1, 1)
        
        # bottleneck conv5-1
        self.conv5_1_1x1_reduce = BatchNormRelu3D(256, 128, 1, 1)
        self.conv5_1_3x3 = BatchNormRelu3D(128, 128, 3, 1, padding=4, dilation=4)
        self.conv5_1_1x1_increase = BatchNorm3D(128, 512, 1, 1)
        
        # conv5_2
        #self.conv5_2_bottleneck = BottleNeck(256, 64, dilation=4)
        self.conv5_2_1x1_reduce = BatchNormRelu3D(512, 128, 1, 1)
        self.conv5_2_3x3 = BatchNormRelu3D(128, 128, 3, 1, padding=4, dilation=4)
        self.conv5_2_1x1_increase = BatchNorm3D(128, 512, 1, 1)
        
        # conv5_3
        #self.conv5_3_bottleneck = BottleNeck(256, 64, dilation=4)
        self.conv5_3_1x1_reduce = BatchNormRelu3D(512, 128, 1, 1)
        self.conv5_3_3x3 = BatchNormRelu3D(128, 128, 3, 1, padding=4, dilation=4)
        self.conv5_3_1x1_increase = BatchNorm3D(128, 512, 1, 1)
        
        # Global average pooling here
        
        # conv5_3_sum
        self.conv5_4_k1 = BatchNorm3D(512, 128, 1, 1)
        self.conv_sub4 = BatchNorm3D(128, 64, 3, stride=2, padding=2, dilation=2) # dilation rate = 2
        self.conv3_1_sub2_proj = BatchNorm3D(128, 64, 1, stride=1)
        self.conv_sub2 = BatchNorm3D(64, 64, 3, stride=2, padding=2, dilation=2) # dilation rate = 2
        
        self.conv_sub2_upsample_adj = BatchNorm3D(64, 64, 1, 1)
        
        # Convolutions for 1 image resolution
        self.conv1_sub1 = BatchNormRelu3D(1, 16, 3, stride=1, padding=1) # stride = 2
        self.conv2_sub1 = BatchNormRelu3D(16, 16, 3, stride=1, padding=1) # stride = 2
        self.conv3_sub1 = BatchNormRelu3D(16, 32, 3, stride=1, padding=1) # stride = 2
        self.conv3_sub1_proj = BatchNorm3D(32, 64, 1, stride=2)
        
        self.sub12_sum_transpose = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        
        self.aux_1_conv = nn.Conv3d(128, self.num_classes, 1, 1)
        self.aux_2_conv = nn.Conv3d(64, self.num_classes, 1, 1)
        self.classification = nn.Conv3d(64, self.num_classes, 1, 1)
>>>>>>> master
        
        self.sigmoid = nn.Softmax(dim=1)
        self.aux_1_sigmoid = nn.Softmax(dim=1)
        self.aux_2_sigmoid = nn.Softmax(dim=1)
        
        # define loss function (self.loss = ...)
        
    # Input x: (shape = (frames, height, width))
    def forward(self, x):
        #print(x.shape)
        l, h, w = x.shape[2:] 

        # image resolution: 1/2 - L, H, W -> L/2, H/2, W/2
        data_sub2 = F.interpolate(x, scale_factor = 1/2, mode="trilinear", align_corners=True)

        # L/2, H/2, W/2 -> L/4, H/4, W/4
        conv1_1_3x3_s2 = self.conv1_1_3x3_s2(data_sub2)
        conv1_2_3x3 = self.conv1_2_3x3(conv1_1_3x3_s2)
        conv1_3_3x3 = self.conv1_3_3x3(conv1_2_3x3)

        # L/4, H/4, W/4 -> L/8, H/8, W/8
        pool1_3x3_s2 = self.pool1_3x3_s2(conv1_3_3x3)

        # L/8, H/8, W/8 -> L/16, H/16, W/16
        conv2_1_1x1_proj = self.conv2_1_1x1_proj(pool1_3x3_s2)
        
        # bottleneck
        conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_3x3 = self.conv2_1_3x3(conv2_1_1x1_reduce)
        conv2_1_1x1_increase = self.conv2_1_1x1_increase(conv2_1_3x3)
        
        # Add residual of conv2_1_1x1 projection to bottleneck of conv2_1
        conv2_1 = F.relu(conv2_1_1x1_proj + conv2_1_1x1_increase)
        conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(conv2_1)
        conv2_2_3x3 = self.conv2_2_3x3(conv2_2_1x1_reduce)
        conv2_2_1x1_increase = self.conv2_2_1x1_increase(conv2_2_3x3)
        
        conv2_2 = F.relu(conv2_1 + conv2_2_1x1_increase)
        conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(conv2_2)
        conv2_3_3x3 = self.conv2_3_3x3(conv2_3_1x1_reduce)
        conv2_3_1x1_increase = self.conv2_3_1x1_increase(conv2_3_3x3)
        
        conv2_3 = F.relu(conv2_2 + conv2_3_1x1_increase)
        
        conv3_1_1x1_proj = self.conv3_1_1x1_proj(conv2_3)
        
        # bottleneck residual block on conv2_3
        conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(conv2_3)
        conv3_1_3x3 = self.conv3_1_3x3(conv3_1_1x1_reduce)
        conv3_1_1x1_increase = self.conv3_1_1x1_increase(conv3_1_3x3)
        
        #conv3_1 = conv3_1_1x1_proj + conv3_1_1x1_increase
        conv3_1 = F.relu(conv3_1_1x1_proj + conv3_1_1x1_increase)
        
        # image resolution: 1/4 - L/16, H/16, W/16 -> L/32, H/32, W/32
        conv3_1_sub4 = F.interpolate(conv3_1, scale_factor=1/2, mode="trilinear", align_corners=True)
        
        #print("conv3_1 - x_sub4 shape:", x_sub4.shape)
        
        conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(conv3_1_sub4)
        conv3_2_3x3 = self.conv3_2_3x3(conv3_2_1x1_reduce)
        conv3_2_1x1_increase = self.conv3_2_1x1_increase(conv3_2_3x3)
        
        conv3_2 = F.relu(conv3_1_sub4 + conv3_2_1x1_increase)
        conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(conv3_2)
        conv3_3_3x3 = self.conv3_3_3x3(conv3_3_1x1_reduce)
        conv3_3_1x1_increase = self.conv3_3_1x1_increase(conv3_3_3x3)
        
        conv3_3 = F.relu(conv3_2 + conv3_3_1x1_increase)
        conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(conv3_3)
        conv3_4_3x3 = self.conv3_4_3x3(conv3_4_1x1_reduce)
        conv3_4_1x1_increase = self.conv3_4_1x1_increase(conv3_4_3x3)
        
        conv3_4 = F.relu(conv3_3 + conv3_4_1x1_increase)
        
        # increase projection
        conv4_1_1x1_proj = self.conv4_1_1x1_proj(conv3_4)
        
        # bottlenecks
        conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(conv3_4)
        conv4_1_3x3 = self.conv4_1_3x3(conv4_1_1x1_reduce)
        conv4_1_1x1_increase = self.conv4_1_1x1_increase(conv4_1_3x3)
        
        conv4_1 = F.relu(conv4_1_1x1_proj + conv4_1_1x1_increase)
        #conv4_1 = conv4_1_1x1_proj + conv4_1_1x1_increase
        conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(conv4_1)
        conv4_2_3x3 = self.conv4_2_3x3(conv4_2_1x1_reduce)
        conv4_2_1x1_increase = self.conv4_2_1x1_increase(conv4_2_3x3)
        
        conv4_2 = F.relu(conv4_1 + conv4_2_1x1_increase)
        conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(conv4_2)
        conv4_3_3x3 = self.conv4_3_3x3(conv4_3_1x1_reduce)
        conv4_3_1x1_increase = self.conv4_3_1x1_increase(conv4_3_3x3)
        
        conv4_3 = F.relu(conv4_2 + conv4_3_1x1_increase)
        conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(conv4_3)
        conv4_4_3x3 = self.conv4_4_3x3(conv4_4_1x1_reduce)
        conv4_4_1x1_increase = self.conv4_4_1x1_increase(conv4_4_3x3)
        
        conv4_4 = F.relu(conv4_3 + conv4_4_1x1_increase)
        conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(conv4_4)
        conv4_5_3x3 = self.conv4_5_3x3(conv4_5_1x1_reduce)
        conv4_5_1x1_increase = self.conv4_5_1x1_increase(conv4_5_3x3)
        
        conv4_5 = F.relu(conv4_4 + conv4_5_1x1_increase)
        conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(conv4_5)
        conv4_6_3x3 = self.conv4_6_3x3(conv4_6_1x1_reduce)
        conv4_6_1x1_increase = self.conv4_6_1x1_increase(conv4_6_3x3)
        
        conv4_6 = F.relu(conv4_5 + conv4_6_1x1_increase)
        
        # increase projection
        conv5_1_1x1_proj = self.conv5_1_1x1_proj(conv4_6)
        
        conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(conv4_6)
        conv5_1_3x3 = self.conv5_1_3x3(conv5_1_1x1_reduce)
        conv5_1_1x1_increase = self.conv5_1_1x1_increase(conv5_1_3x3)
        
        conv5_1 = F.relu(conv5_1_1x1_proj + conv5_1_1x1_increase)
        conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(conv5_1)
        conv5_2_3x3 = self.conv5_2_3x3(conv5_2_1x1_reduce)
        conv5_2_1x1_increase = self.conv5_2_1x1_increase(conv5_2_3x3)
        
        conv5_2 = F.relu(conv5_1 + conv5_2_1x1_increase)
        conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(conv5_2)
        conv5_3_3x3 = self.conv5_3_3x3(conv5_3_1x1_reduce)
        conv5_3_1x1_increase = self.conv5_3_1x1_increase(conv5_3_3x3)
        
        conv5_3 = F.relu(conv5_2 + conv5_3_1x1_increase)
        
        # Do global average pooling here (pyramid pooling module)
        #print("conv5_3 shape: ", x_sub4.shape)
        l, w, h = conv5_3.shape[2:]
        l = int(l)
        w = int(w)
        h = int(h)
        pool1 = F.avg_pool3d(conv5_3, (l, w, h), stride=(l, w, h))
        pool1 = F.interpolate(pool1, size=[l, w, h], mode='trilinear', align_corners=True) # resize pool1
        
        pool2 = F.avg_pool3d(conv5_3, (int(l/2), int(w/2), int(h/2)), stride=(int(l/2), int(w/2), int(l/2)))
        pool2 = F.interpolate(pool2, size=[l, w, h], mode='trilinear', align_corners=True) # resize pool2
        
        pool3 = F.avg_pool3d(conv5_3, (int(l/3), int(w/3), int(h/3)), stride=(int(l/3), int(w/3), int(h/3)))
        pool3 = F.interpolate(pool3, size=(l, w, h), mode='trilinear', align_corners=True) # resize pool3
        
        pool6 = F.avg_pool3d(conv5_3, (int(l/4), int(w/4), int(h/4)), stride=(int(l/4), int(w/4), int(h/4)))
        pool6 = F.interpolate(pool6, size=(l, w, h), mode='trilinear', align_corners=True)
        
        conv5_3_sum = conv5_3 + pool1 + pool2 + pool3 + pool6#add pooling layers and x_sub5
        
        #print("conv5_3_sum shape:", conv5_3_sum.shape)
        
        conv5_4_k1 = self.conv5_4_k1(conv5_3_sum)  # apply Batch Normalization to addition layer
        
<<<<<<< HEAD
        # perform interpolation by scale factor 2 on output
        conv5_4_interp = F.interpolate(conv5_4_k1, scale_factor=4, mode='trilinear', align_corners=True) 
        conv_sub4 = self.conv_sub4(conv5_4_interp)
        
        # perform convolution on third layer sum
        #conv3_1 = F.interpolate(conv3_1, scale_factor=2, mode='trilinear', align_corners=True)
        conv3_1_sub2_proj = self.conv3_1_sub2_proj(conv3_1)
        #print("conv3_1_sub2_proj:", conv3_1_sub2_proj.shape)
        #print("conv_sub4:", conv_sub4.shape)
        sub24_sum = conv3_1_sub2_proj + conv_sub4 # add outputs from two layers
        sub24_sum_interp = F.interpolate(sub24_sum, scale_factor=4, mode='trilinear', align_corners=True)
        conv_sub2 = self.conv_sub2(sub24_sum_interp)
       
        # image resolution: 1
        #print("x:", x.shape)
        conv1_sub1 = self.conv1_sub1(x)
        #print("conv1_sub1:", conv1_sub1.shape)
        conv2_sub1 = self.conv2_sub1(conv1_sub1)
        #print("conv2_sub1:", conv2_sub1.shape)
        conv3_sub1 = self.conv3_sub1(conv2_sub1)
        #print("conv3_sub1:", conv3_sub1.shape)
        conv3_sub1_proj = self.conv3_sub1_proj(conv3_sub1)
        #print("conv_sub2:", conv_sub2.shape)
        #print("conv3_sub1_proj:", conv3_sub1_proj.shape)
        #sub12_sum = F.relu(conv_sub2 + conv3_sub1_proj, inplace=True)
       
        sub12_sum = conv_sub2 + conv3_sub1_proj
=======
        # perform interpolation by scale factor 4 on output 
        conv5_4_interp = F.interpolate(conv5_4_k1, scale_factor=8, mode='trilinear', align_corners=True) 
        conv_sub4 = self.conv_sub4(conv5_4_interp)
        print("conv5_4_interp:", conv5_4_interp.shape)
        
        # perform convolution on third layer sum
        conv3_1 = F.interpolate(conv3_1, scale_factor=2, mode='trilinear', align_corners=True)
        conv3_1_sub2_proj = self.conv3_1_sub2_proj(conv3_1)
        
        print("conv3_1_sub2_proj:", conv3_1_sub2_proj.shape)
        print("conv_sub4:", conv_sub4.shape)
        
        sub24_sum = conv3_1_sub2_proj + conv_sub4 # add outputs from two layers
        sub24_sum_interp = F.interpolate(sub24_sum, scale_factor=8, mode='trilinear', align_corners=True)
        
        conv_sub2 = self.conv_sub2(sub24_sum_interp)
        conv_sub2_upsample = F.interpolate(conv_sub2, scale_factor=4, mode='trilinear', align_corners=True)
        conv_sub2_upsample = self.conv_sub2_upsample_adj(conv_sub2_upsample) # 32 -> 128
        print("conv_sub2_upsample:", conv_sub2_upsample.shape) 
        
        # image resolution: 1
        print("x:", x.shape)
        conv1_sub1 = self.conv1_sub1(x)
        
        print("conv1_sub1:", conv1_sub1.shape)
        conv2_sub1 = self.conv2_sub1(conv1_sub1)
        
        print("conv2_sub1:", conv2_sub1.shape)
        conv3_sub1 = self.conv3_sub1(conv2_sub1)
        
        print("conv3_sub1:", conv3_sub1.shape)
        conv3_sub1_proj = self.conv3_sub1_proj(conv3_sub1)
        
        print("conv_sub2:", conv_sub2.shape)
        print("conv3_sub1_proj:", conv3_sub1_proj.shape)
        #sub12_sum = F.relu(conv_sub2 + conv3_sub1_proj, inplace=True)
       
        sub12_sum = conv_sub2_upsample + conv3_sub1_proj
>>>>>>> master
        #print(sub12_sum.shape)
        sub12_sum_transpose = self.sub12_sum_transpose(sub12_sum)
        #print(sub12_sum_transpose.shape)
        #sub12_sum_interp = F.interpolate(sub12_sum, scale_factor=4, mode='trilinear', align_corners=True)
        
        # 1 image resolution
        conv6_cls = self.classification(sub12_sum_transpose) # apply classification convolution
        #print("1:", conv6_cls)
        conv6_cls = self.sigmoid(conv6_cls) # apply softmax layer
   
        if self.training:
            # 1/8 image resolution
            sub4_out = self.aux_1_conv(conv5_4_interp)
            #print("aux_1:", sub4_out)
            sub4_out = self.aux_1_sigmoid(sub4_out) # apply softmax
            
            # 1/4 image resolution
            sub24_out = self.aux_2_conv(sub24_sum_interp)
            #print("aux_2:", sub24_out)
            sub24_out = self.aux_2_sigmoid(sub24_out)
            
            return (conv6_cls, sub24_out, sub4_out)
        else:
            return (conv6_cls)
<<<<<<< HEAD

=======
        
>>>>>>> master
class FullResolutionICNet(nn.Module):
    def __init__(self, num_classes):
        super(FullResolutionICNet, self).__init__()
        # Initialize instance variables
        self.num_classes = num_classes
        
        #Conv3d function: 
        # x = conv3d (x, 1, 8) # 256
        # x = conv3d (x, 8, 8) # 256
        # x = conv3d (x, 8, 8) # 256
        # x = conv3d (x, 8, 16, stride = 2) or what ever downsample, pooling # 256 ->128

        # 2_featmap_up = upsample (2_featmap, scale = 4) # 32->128
        # 2_featmap_adj = conv3d (2_featmap_up, num of filter size last model, 16) # 128
        # x += 2_featmap_adj # 128
        # x = conv3d (x, 16, 16) # 128

        # x = ConvTranspose3d (x, 16, 8, stride = 2) # 128->256
        # x = conv3d (x, 8, 3) #256 * 3 classes
        # x = Softmax (x, dim=1)
        
        #torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,   padding_mode='zeros')
        
       # - More Notes on the input variables:
       #      - stride controls the stride for the cross-correlation.
       #      - padding controls the amount of implicit zero-paddings on both sides for padding number of points for each  dimension.
       #      - dilation controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this link has a nice visualization of what dilation does.
       #     - groups controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups. 
       #         - At groups=1, all inputs are convolved to all outputs.
       #         - At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels, and producing half the output channels, and both subsequently concatenated. 
       #         - At groups= in_channels, each input channel is convolved with its own set of filters, of size [out_channels/in_channels]
        
        
        # BatchNormRelu3D and BatchNorm3D(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=False)
        
        # Convolutions for 1/2 Resolution of Image
        
        # Do trilinear interpolation to get 1/2 resolution
        # s2 = stride 2
        self.conv1_1_3x3_s2 = BatchNormRelu3D(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv1_2_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv1_3_3x3 = BatchNormRelu3D(8, 16,  3, 1, padding=1)
        
        # max pooling does not affect the number of out channels, but rather reduces the spatial dimensions of the output for simplicity in the model
        self.pool1_3x3_s2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Apply these convolutions to the max pooling layer
        self.conv2_1_1x1_proj = BatchNorm3D(16, 32, 1, 1)
        
        #self.conv2_1_bottleneck = BottleNeck(128, 32, 1) 
        # bottleneck
        self.conv2_1_1x1_reduce = BatchNormRelu3D(16, 8, 1, 1)
        self.conv2_1_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv2_1_1x1_increase = BatchNorm3D(8, 32, 1, 1)
        
        # conv2_2
        #self.conv2_2_bottleneck = BottleNeck(32, 8, dilation=1) # bottleneck
        self.conv2_2_1x1_reduce = BatchNormRelu3D(32, 8, 1, 1)
        self.conv2_2_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv2_2_1x1_increase = BatchNorm3D(8, 32, 1, 1)
        
        # conv2_3
        #self.conv2_3_bottleneck = BottleNeck(32, 8, dilation=1)
        self.conv2_3_1x1_reduce = BatchNormRelu3D(32, 8, 1, 1)
        self.conv2_3_3x3 = BatchNormRelu3D(8, 8, 3, 1, padding=1)
        self.conv2_3_1x1_increase = BatchNorm3D(8, 32, 1, 1)
        
        # conv3_1
        self.conv3_1_1x1_proj = BatchNorm3D(32, 64, 1, stride=2) # stride 2
        
        # bottleneck building block
        self.conv3_1_1x1_reduce = BatchNormRelu3D(32, 16, 1, 2) # stride 2
        self.conv3_1_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_1_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # self.conv3_1_bottleneck = BottleNeckConv3_1(256, 64, 1)
        
        # Convolutions for 1/4 Image Resolution
        
        # conv3_2 - bottleneck building block
        #self.conv3_2_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_2_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv3_2_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_2_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv3_3 - bottleneck building block
        #self.conv3_3_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_3_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv3_3_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_3_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv3_4
        #self.conv3_4_bottleneck = BottleNeck(64, 16, dilation=1)
        self.conv3_4_1x1_reduce = BatchNormRelu3D(64, 16, 1, 1)
        self.conv3_4_3x3 = BatchNormRelu3D(16, 16, 3, 1, padding=1)
        self.conv3_4_1x1_increase = BatchNorm3D(16, 64, 1, 1)
        
        # conv4_1
        self.conv4_1_1x1_proj = BatchNorm3D(64, 128, 1, 1)
        
        #self.conv4_1_bottleneck = BottleNeck(512, 128, dilation=2)
        self.conv4_1_1x1_reduce = BatchNormRelu3D(64, 32, 1, 1)
        self.conv4_1_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_1_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_2
        #self.conv4_2_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_2_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_2_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_2_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_3
        #self.conv4_3_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_3_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_3_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_3_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_4
        #self.conv4_4_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_4_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_4_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_4_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_5
        #self.conv4_5_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_5_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_5_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_5_1x1_increase = BatchNorm3D(32, 128, 1, 1)
        
        # conv4_6
        #self.conv4_6_bottleneck = BottleNeck(128, 32, dilation=2)
        self.conv4_6_1x1_reduce = BatchNormRelu3D(128, 32, 1, 1)
        self.conv4_6_3x3 = BatchNormRelu3D(32, 32, 3, 1, padding=2, dilation=2)
        self.conv4_6_1x1_increase = BatchNorm3D(32, 128, 1, 1)
    
        # conv5_1
        self.conv5_1_1x1_proj = BatchNorm3D(128, 256, 1, 1)
        
        # bottleneck conv5-1
        self.conv5_1_1x1_reduce = BatchNormRelu3D(128, 64, 1, 1)
        self.conv5_1_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_1_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv5_2
        #self.conv5_2_bottleneck = BottleNeck(256, 64, dilation=4)
        self.conv5_2_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv5_2_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_2_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # conv5_3
        #self.conv5_3_bottleneck = BottleNeck(256, 64, dilation=4)
        self.conv5_3_1x1_reduce = BatchNormRelu3D(256, 64, 1, 1)
        self.conv5_3_3x3 = BatchNormRelu3D(64, 64, 3, 1, padding=4, dilation=4)
        self.conv5_3_1x1_increase = BatchNorm3D(64, 256, 1, 1)
        
        # Global average pooling here
        
        # conv5_3_sum
        self.conv5_4_k1 = BatchNorm3D(256, 64, 1, 1)
<<<<<<< HEAD
        self.conv_sub4 = BatchNorm3D(64, 32, 3, stride=2, padding=2, dilation=2) # dilation rate = 2
        self.conv3_1_sub2_proj = BatchNorm3D(64, 32, 1, stride=1)
        self.conv_sub2 = BatchNorm3D(32, 32, 3, stride=2, padding=2, dilation=2) # dilation rate = 2
        
        # Convolutions for 1 image resolution
        self.conv1_sub1 = BatchNormRelu3D(1, 8, 3, stride=1, padding=1) # stride = 2
        self.conv2_sub1 = BatchNormRelu3D(8, 8, 3, stride=1, padding=1) # stride = 2
        self.conv3_sub1 = BatchNormRelu3D(8, 16, 3, stride=2, padding=1) # stride = 2
        self.conv3_sub1_proj = BatchNorm3D(16, 32, 1, 1)
=======
        self.conv_sub4 = BatchNorm3D(64, 32, 3, stride=4, padding=2, dilation=2) # dilation rate = 2
        self.conv3_1_sub2_proj = BatchNorm3D(64, 32, 1, stride=1)
        self.conv_sub2 = BatchNorm3D(32, 32, 3, stride=2, padding=2, dilation=2) # dilation rate = 2
        
        self.conv_sub2_upsample_adj = BatchNormRelu3D(32, 32, 1, 1)
        
        # Convolutions for 1 image resolution
        self.conv1_sub1 = BatchNormRelu3D(1, 8, 3, stride=1, padding=1) # stride = 2
        self.conv2_sub1 = BatchNormRelu3D(8, 8, 3, stride=1, padding=1) # stride = 2
        self.conv3_sub1 = BatchNormRelu3D(8, 16, 3, stride=1, padding=1) # stride = 2
        self.conv3_sub1_proj = BatchNorm3D(16, 32, 1, stride=2)
>>>>>>> master
        
        self.sub12_sum_transpose = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2)
        
        self.aux_1_conv = nn.Conv3d(64, self.num_classes, 1, 1)
        self.aux_2_conv = nn.Conv3d(32, self.num_classes, 1, 1)
        self.classification = nn.Conv3d(32, self.num_classes, 1, 1)
        
        self.sigmoid = nn.Softmax(dim=1)
        self.aux_1_sigmoid = nn.Softmax(dim=1)
        self.aux_2_sigmoid = nn.Softmax(dim=1)
        
        # define loss function (self.loss = ...)
        
    # Input x: (shape = (frames, height, width))
    def forward(self, x):
        #print(x.shape)
        l, h, w = x.shape[2:] 

        # image resolution: 1/2 - L, H, W -> L/2, H/2, W/2
        data_sub2 = F.interpolate(x, scale_factor = 1/2, mode="trilinear", align_corners=True)

        # L/2, H/2, W/2 -> L/4, H/4, W/4
        conv1_1_3x3_s2 = self.conv1_1_3x3_s2(data_sub2)
        conv1_2_3x3 = self.conv1_2_3x3(conv1_1_3x3_s2)
        conv1_3_3x3 = self.conv1_3_3x3(conv1_2_3x3)

        # L/4, H/4, W/4 -> L/8, H/8, W/8
        pool1_3x3_s2 = self.pool1_3x3_s2(conv1_3_3x3)

        # L/8, H/8, W/8 -> L/16, H/16, W/16
        conv2_1_1x1_proj = self.conv2_1_1x1_proj(pool1_3x3_s2)
        
        # bottleneck
        conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_3x3 = self.conv2_1_3x3(conv2_1_1x1_reduce)
        conv2_1_1x1_increase = self.conv2_1_1x1_increase(conv2_1_3x3)
        
        # Add residual of conv2_1_1x1 projection to bottleneck of conv2_1
        conv2_1 = F.relu(conv2_1_1x1_proj + conv2_1_1x1_increase)
        conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(conv2_1)
        conv2_2_3x3 = self.conv2_2_3x3(conv2_2_1x1_reduce)
        conv2_2_1x1_increase = self.conv2_2_1x1_increase(conv2_2_3x3)
        
        conv2_2 = F.relu(conv2_1 + conv2_2_1x1_increase)
        conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(conv2_2)
        conv2_3_3x3 = self.conv2_3_3x3(conv2_3_1x1_reduce)
        conv2_3_1x1_increase = self.conv2_3_1x1_increase(conv2_3_3x3)
        
        conv2_3 = F.relu(conv2_2 + conv2_3_1x1_increase)
        
        conv3_1_1x1_proj = self.conv3_1_1x1_proj(conv2_3)
        
        # bottleneck residual block on conv2_3
        conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(conv2_3)
        conv3_1_3x3 = self.conv3_1_3x3(conv3_1_1x1_reduce)
        conv3_1_1x1_increase = self.conv3_1_1x1_increase(conv3_1_3x3)
        
        #conv3_1 = conv3_1_1x1_proj + conv3_1_1x1_increase
        conv3_1 = F.relu(conv3_1_1x1_proj + conv3_1_1x1_increase)
        
        # image resolution: 1/4 - L/16, H/16, W/16 -> L/32, H/32, W/32
        conv3_1_sub4 = F.interpolate(conv3_1, scale_factor=1/2, mode="trilinear", align_corners=True)
        
        #print("conv3_1 - x_sub4 shape:", x_sub4.shape)
        
        conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(conv3_1_sub4)
        conv3_2_3x3 = self.conv3_2_3x3(conv3_2_1x1_reduce)
        conv3_2_1x1_increase = self.conv3_2_1x1_increase(conv3_2_3x3)
        
        conv3_2 = F.relu(conv3_1_sub4 + conv3_2_1x1_increase)
        conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(conv3_2)
        conv3_3_3x3 = self.conv3_3_3x3(conv3_3_1x1_reduce)
        conv3_3_1x1_increase = self.conv3_3_1x1_increase(conv3_3_3x3)
        
        conv3_3 = F.relu(conv3_2 + conv3_3_1x1_increase)
        conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(conv3_3)
        conv3_4_3x3 = self.conv3_4_3x3(conv3_4_1x1_reduce)
        conv3_4_1x1_increase = self.conv3_4_1x1_increase(conv3_4_3x3)
        
        conv3_4 = F.relu(conv3_3 + conv3_4_1x1_increase)
        
        # increase projection
        conv4_1_1x1_proj = self.conv4_1_1x1_proj(conv3_4)
        
        # bottlenecks
        conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(conv3_4)
        conv4_1_3x3 = self.conv4_1_3x3(conv4_1_1x1_reduce)
        conv4_1_1x1_increase = self.conv4_1_1x1_increase(conv4_1_3x3)
        
        conv4_1 = F.relu(conv4_1_1x1_proj + conv4_1_1x1_increase)
        #conv4_1 = conv4_1_1x1_proj + conv4_1_1x1_increase
        conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(conv4_1)
        conv4_2_3x3 = self.conv4_2_3x3(conv4_2_1x1_reduce)
        conv4_2_1x1_increase = self.conv4_2_1x1_increase(conv4_2_3x3)
        
        conv4_2 = F.relu(conv4_1 + conv4_2_1x1_increase)
        conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(conv4_2)
        conv4_3_3x3 = self.conv4_3_3x3(conv4_3_1x1_reduce)
        conv4_3_1x1_increase = self.conv4_3_1x1_increase(conv4_3_3x3)
        
        conv4_3 = F.relu(conv4_2 + conv4_3_1x1_increase)
        conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(conv4_3)
        conv4_4_3x3 = self.conv4_4_3x3(conv4_4_1x1_reduce)
        conv4_4_1x1_increase = self.conv4_4_1x1_increase(conv4_4_3x3)
        
        conv4_4 = F.relu(conv4_3 + conv4_4_1x1_increase)
        conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(conv4_4)
        conv4_5_3x3 = self.conv4_5_3x3(conv4_5_1x1_reduce)
        conv4_5_1x1_increase = self.conv4_5_1x1_increase(conv4_5_3x3)
        
        conv4_5 = F.relu(conv4_4 + conv4_5_1x1_increase)
        conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(conv4_5)
        conv4_6_3x3 = self.conv4_6_3x3(conv4_6_1x1_reduce)
        conv4_6_1x1_increase = self.conv4_6_1x1_increase(conv4_6_3x3)
        
        conv4_6 = F.relu(conv4_5 + conv4_6_1x1_increase)
        
        # increase projection
        conv5_1_1x1_proj = self.conv5_1_1x1_proj(conv4_6)
        
        conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(conv4_6)
        conv5_1_3x3 = self.conv5_1_3x3(conv5_1_1x1_reduce)
        conv5_1_1x1_increase = self.conv5_1_1x1_increase(conv5_1_3x3)
        
        conv5_1 = F.relu(conv5_1_1x1_proj + conv5_1_1x1_increase)
        conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(conv5_1)
        conv5_2_3x3 = self.conv5_2_3x3(conv5_2_1x1_reduce)
        conv5_2_1x1_increase = self.conv5_2_1x1_increase(conv5_2_3x3)
        
        conv5_2 = F.relu(conv5_1 + conv5_2_1x1_increase)
        conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(conv5_2)
        conv5_3_3x3 = self.conv5_3_3x3(conv5_3_1x1_reduce)
        conv5_3_1x1_increase = self.conv5_3_1x1_increase(conv5_3_3x3)
        
        conv5_3 = F.relu(conv5_2 + conv5_3_1x1_increase)
        
        # Do global average pooling here (pyramid pooling module)
        #print("conv5_3 shape: ", x_sub4.shape)
        l, w, h = conv5_3.shape[2:]
        l = int(l)
        w = int(w)
        h = int(h)
        pool1 = F.avg_pool3d(conv5_3, (l, w, h), stride=(l, w, h))
        pool1 = F.interpolate(pool1, size=[l, w, h], mode='trilinear', align_corners=True) # resize pool1
        
        pool2 = F.avg_pool3d(conv5_3, (int(l/2), int(w/2), int(h/2)), stride=(int(l/2), int(w/2), int(l/2)))
        pool2 = F.interpolate(pool2, size=[l, w, h], mode='trilinear', align_corners=True) # resize pool2
        
        pool3 = F.avg_pool3d(conv5_3, (int(l/3), int(w/3), int(h/3)), stride=(int(l/3), int(w/3), int(h/3)))
        pool3 = F.interpolate(pool3, size=(l, w, h), mode='trilinear', align_corners=True) # resize pool3
        
        pool6 = F.avg_pool3d(conv5_3, (int(l/4), int(w/4), int(h/4)), stride=(int(l/4), int(w/4), int(h/4)))
        pool6 = F.interpolate(pool6, size=(l, w, h), mode='trilinear', align_corners=True)
        
        conv5_3_sum = conv5_3 + pool1 + pool2 + pool3 + pool6#add pooling layers and x_sub5
        
        #print("conv5_3_sum shape:", conv5_3_sum.shape)
        
        conv5_4_k1 = self.conv5_4_k1(conv5_3_sum)  # apply Batch Normalization to addition layer
        
<<<<<<< HEAD
        # perform interpolation by scale factor 4 on output (32 -> 128)
        conv5_4_interp = F.interpolate(conv5_4_k1, scale_factor=4, mode='trilinear', align_corners=True) 
        conv_sub4 = self.conv_sub4(conv5_4_interp)
        print("conv5_4_interp:", conv5_4_interp.shape)
=======
        # perform interpolation by scale factor 4 on output 
        conv5_4_interp = F.interpolate(conv5_4_k1, scale_factor=8, mode='trilinear', align_corners=True) 
        conv_sub4 = self.conv_sub4(conv5_4_interp)
        #print("conv5_4_interp:", conv5_4_interp.shape)
>>>>>>> master
        
        # perform convolution on third layer sum
        #conv3_1 = F.interpolate(conv3_1, scale_factor=2, mode='trilinear', align_corners=True)
        conv3_1_sub2_proj = self.conv3_1_sub2_proj(conv3_1)
<<<<<<< HEAD
        
        print("conv3_1_sub2_proj:", conv3_1_sub2_proj.shape)
        print("conv_sub4:", conv_sub4.shape)
        
        sub24_sum = conv3_1_sub2_proj + conv_sub4 # add outputs from two layers
        sub24_sum_interp = F.interpolate(sub24_sum, scale_factor=4, mode='trilinear', align_corners=True)
        conv_sub2 = self.conv_sub2(sub24_sum_interp)
        conv_sub2_upsample = F.interpolate(conv_sub2, scale_factor=4, mode='trilinear', align_corners=True)
        print("conv_sub2_upsample:", conv_sub2_upsample.shape)
        
        # image resolution: 1
        print("x:", x.shape)
        conv1_sub1 = self.conv1_sub1(x)
        
        print("conv1_sub1:", conv1_sub1.shape)
        conv2_sub1 = self.conv2_sub1(conv1_sub1)
        
        print("conv2_sub1:", conv2_sub1.shape)
        conv3_sub1 = self.conv3_sub1(conv2_sub1)
        
        print("conv3_sub1:", conv3_sub1.shape)
        conv3_sub1_proj = self.conv3_sub1_proj(conv3_sub1)
        
        print("conv_sub2:", conv_sub2.shape)
        print("conv3_sub1_proj:", conv3_sub1_proj.shape)
=======
        #conv3_1_sub2_proj = F.interpolate(conv3_1_sub2_proj, scale_factor=2, mode='trilinear', align_corners=True)
        
        #print("conv3_1_sub2_proj:", conv3_1_sub2_proj.shape)
        #print("conv_sub4:", conv_sub4.shape)
        
        sub24_sum = conv3_1_sub2_proj + conv_sub4 # add outputs from two layers
        sub24_sum_interp = F.interpolate(sub24_sum, scale_factor=8, mode='trilinear', align_corners=True)
        
        conv_sub2 = self.conv_sub2(sub24_sum_interp)
        #print("conv_sub2:", conv_sub2.shape)
        conv_sub2_upsample = F.interpolate(conv_sub2, scale_factor=2, mode='trilinear', align_corners=True)
        conv_sub2_upsample = self.conv_sub2_upsample_adj(conv_sub2_upsample) # 32 -> 128
        #print("conv_sub2_upsample:", conv_sub2_upsample.shape) 
        
        # image resolution: 1
        #print("x:", x.shape)
        conv1_sub1 = self.conv1_sub1(x)
        
        #print("conv1_sub1:", conv1_sub1.shape)
        conv2_sub1 = self.conv2_sub1(conv1_sub1)
        
        #print("conv2_sub1:", conv2_sub1.shape)
        conv3_sub1 = self.conv3_sub1(conv2_sub1)
        
        #print("conv3_sub1:", conv3_sub1.shape)
        conv3_sub1_proj = self.conv3_sub1_proj(conv3_sub1)
        
        #print("conv_sub2:", conv_sub2.shape)
        #print("conv3_sub1_proj:", conv3_sub1_proj.shape)
>>>>>>> master
        #sub12_sum = F.relu(conv_sub2 + conv3_sub1_proj, inplace=True)
       
        sub12_sum = conv_sub2_upsample + conv3_sub1_proj
        #print(sub12_sum.shape)
        sub12_sum_transpose = self.sub12_sum_transpose(sub12_sum)
        #print(sub12_sum_transpose.shape)
        #sub12_sum_interp = F.interpolate(sub12_sum, scale_factor=4, mode='trilinear', align_corners=True)
        
        # 1 image resolution
        conv6_cls = self.classification(sub12_sum_transpose) # apply classification convolution
        #print("1:", conv6_cls)
        conv6_cls = self.sigmoid(conv6_cls) # apply softmax layer
   
        if self.training:
            # 1/8 image resolution
            sub4_out = self.aux_1_conv(conv5_4_interp)
            #print("aux_1:", sub4_out)
            sub4_out = self.aux_1_sigmoid(sub4_out) # apply softmax
            
            # 1/4 image resolution
            sub24_out = self.aux_2_conv(sub24_sum_interp)
            #print("aux_2:", sub24_out)
            sub24_out = self.aux_2_sigmoid(sub24_out)
            
            return (conv6_cls, sub24_out, sub4_out)
        else:
<<<<<<< HEAD
            return (conv6_cls)
=======
            return (conv6_cls)
>>>>>>> master
