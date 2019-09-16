import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable

"""
Conv3d function: 
        
    torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,   padding_mode='zeros')
        
        - More Notes on the input variables:
             - stride controls the stride for the cross-correlation.
             - padding controls the amount of implicit zero-paddings on both sides for padding number of points for each dimension.
             - dilation controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this link has a nice visualization of what dilation does.
             - groups controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups. 
                - At groups=1, all inputs are convolved to all outputs.
                - At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels, and producing half the output channels, and both subsequently concatenated. 
                - At groups= in_channels, each input channel is convolved with its own set of filters, of size [out_channels/in_channels]
                
                
BatchNorm3d function:

    torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
"""

class BottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, dilation):
        super(BottleNeck, self).__init__()
        
        self.conv_1x1_reduce = BatchNormRelu3D(in_channels, mid_channels, kernel_size=1, stride=1)
        self.conv_3x3 = BatchNormRelu3D(mid_channels, mid_channels, 3, 1, padding=dilation, dilation=dilation)
        self.conv_1x1_increase = BatchNorm3D(mid_channels, in_channels, 1, 1)
        
    def forward(self, x):
        residual = x
        x = self.conv_1x1_reduce(x)
        x = self.conv_3x3(x)
        x = self.conv_1x1_increase(x)
        return F.relu(x + residual, inplace=True)

class ConvRelu3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(ConvRelu3D, self).__init__()
        
        # Use nn.Sequential to make sequence of convolutions + batch normalization and relu
        self.sequenceOfConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # return output from applying sequence of convolutions on input x
        return self.sequenceOfConv(x)
    
class BatchNormReluDropout3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, dropout_rate=0.5, bias=False):
        super(BatchNormReluDropout3D, self).__init__()
        
        # Use nn.Sequential to make sequence of convolutions + batch normalization and relu
        self.sequenceOfConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        
    def forward(self, x):
        # return output from applying sequence of convolutions on input x
        return self.sequenceOfConv(x)
    
class BatchNormRelu3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(BatchNormRelu3D, self).__init__()
        
        # Use nn.Sequential to make sequence of convolutions + batch normalization and relu
        self.sequenceOfConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # return output from applying sequence of convolutions on input x
        return self.sequenceOfConv(x)

class BatchNorm3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=False):
        super(BatchNorm3D, self).__init__()
        # Use nn.Sequential to make sequence of convolutions + batch normalization and relu
        self.sequenceOfConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm3d(out_channels),
        )      
    def forward(self, x):
        # return output from applying sequence of convolutions on input x
        return self.sequenceOfConv(x)
    
def flatten(x):
    return x.reshape(x.shape[0], -1)
    
def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=0, std=1)

def downsample_label(label, scale_factor):
    mode = 'bicubic' if get_dimensions(label) == 4 else 'trilinear' 
    return F.interpolate(label, scale_factor=scale_factor, mode=mode, align_corners=True)

def downsample_input3d(inp, scale_factor):
    return F.interpolate(inp, scale_factor=scale_factor, mode='trilinear', align_corners=True)

def downsample_label_bicubic(label, scale_factor):
    return F.interpolate(label, scale_factor=scale_factor, mode='bicubic', align_corners=True)
        
def get_dimensions(x):
    return len(list(x.size()))

def get_bounding_box_image(image, img_res, size=128):
    centroid = find_centroid_image(image, img_res)
    x = int(centroid[0])
    y = int(centroid[1])
    z = int(centroid[2])
    
    image = image.view(1,1,img_res[0], img_res[1], img_res[2])
        
    half_size = int(size / 2)
    return image[:, :, x-half_size:x+half_size, y-half_size:y+half_size, z-half_size:z+half_size]

def get_bounding_box_bv_label(bv_label, size=128):
    bv_label.squeeze_()
    
    half_size = int(size / 2)
    x, y, z = find_bv_centroid(bv_label)
    return bv_label[x-half_size:x+half_size, y-half_size:y+half_size, z-half_size:z+half_size]

def get_bounding_box_coords_from_centroid(centroid, size=128):
    half_size = int(size / 2)
    x = int(centroid[0])
    y = int(centroid[1])
    z = int(centroid[2])
    # Top Right: (x + 64), (y+64), (z-64)
    coord1 = (x+half_size, y+half_size, z-half_size)
    coord2 = (x+half_size, y+half_size, z+half_size)
    coord3 = (x+half_size, y-half_size, z+half_size)
    coord4 = (x+half_size, y-half_size, z-half_size)
    
    coord5 = (x-half_size, y-half_size, z-half_size)
    coord6 = (x-half_size, y-half_size, z+half_size)
    coord7 = (x-half_size, y+half_size, z+half_size)
    coord8 = (x-half_size, y+half_size, z-half_size)
    
    return (coord1, coord2, coord3, coord4, coord5, coord6, coord7, coord8)

def find_centroid_image(image, image_res):
    centroid = torch.zeros(3)
    x = image_res[0] / 2
    y = image_res[1] / 2
    z = image_res[2] / 2
    centroid[0] = x
    centroid[1] = y
    centroid[2] = z
    return centroid

# Find center of BV based on segmentation map or label
def find_bv_centroid(bv_label):
    # label shape: (1, 256, 256, 256)
    bv_label.squeeze_()
    bv_label = bv_label.to(torch.float32)
    # Ensure that the label is one-hot encoded / binarized
    #if pred == True:
    #    bv_label = binarize_output(bv_label)
    
    one_indices = (bv_label == 1).nonzero().to(torch.float32) # get indices of nonzero points of bv label (value of one, i.e. where BV is present)
    x = one_indices[:, -3].mean()
    y = one_indices[:, -2].mean()
    z = one_indices[:, -1].mean()
    return int(x), int(y), int(z)

def binarize_output(label):
    if torch.is_tensor(label):
        label = label.cpu().detach().numpy()
    binary = torch.from_numpy((label == label.max(axis=1)[:, None]).astype(int))
    return binary

def binarize_output_threshold(label, threshold=0.9):
    label[label >= threshold] = 1
    label[label < threshold] = 0
    return label

def binarize_output2(label):
    label = label.cpu().detach().numpy()
    z = np.zeros_like(label)
    z[np.range(len(label)), label.argmax(1)] = 1
    return z

def pad_zeros(array, x, y, z):
    array = array.cpu().detach().numpy()
    z = np.zeros((x,y,z))
    z[:x, :y] = array
    return z

def load_bbox_bv_label(label, size=128):
    half = int(size/2)
    x,y,z = loadbvcenter(label)
    return label[x-half:x+half, y-half:y+half, z-half:z+half]

def loadbvcenter(img):
	'''
	get the bv center
	'''
	img = img.cpu().detach().numpy()
	bvmask = loadbvmask(img)
    
	x = int(np.mean(bvmask[0:2]))
	y = int(np.mean(bvmask[2:4]))
	z = int(np.mean(bvmask[4:6]))

	return x, y, z

def loadbvmask(img):
    '''
    Truely stupid and brutal force way to Find mask of BV
    '''
    if torch.is_tensor(img):
        img = img.cpu().detach().numpy()
    
    img = (img > 0.66).astype(np.float32)
    # BVmask.shape = 1, X, Y, Z

    _, X, Y, Z = img.shape
    x1, y1, z1 = 0, 0, 0
    x2, y2, z2 = X-1, Y-1, Z-1

    while x1 < X:
        if (np.sum(img[:,x1,:,:]) > 0): # ~take a slice and check!
            break
        else:
            x1 += 1

    while y1 < Y:
        if (np.sum(img[:,:,y1,:]) > 0):
            break
        else:
            y1 += 1

    while z1 < Z:
        if (np.sum(img[:,:,:,z1]) > 0):
            break
        else:
            z1 += 1

    while x2 > 0:
        if (np.sum(img[:,x2,:,:]) > 0): 
            break
        else:
            x2 -= 1

    while y2 > 0:
        if (np.sum(img[:,:,y2,:]) > 0): 
            break
        else:
            y2 -= 1

    while z2 > 0:
        if (np.sum(img[:,:,:,z2]) > 0): 
            break
        else:
            z2 -= 1

    return np.array([x1, x2, y1, y2, z1, z2])