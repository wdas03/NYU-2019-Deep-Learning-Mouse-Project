import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# Implement of 3D version of RefineNet: Multi-Path Refinement 
# Networks for High-Resolution Semantic Segmentation
################################################################################

class residual_conv_unit(nn.Module):

	'''residual conv unit
	
	Parameters
	----------
	in_channel: int, in channel number
	out_channel: int, out channel number

	Callable
	--------
	return out, tensor, outchannel

	'''
	
	def __init__(self, in_channel, out_channel, residual=True):
		super(residual_conv_unit, self).__init__()
		self.concat_factor = out_channel / in_channel        
		self.residual = residual
		self.conv_1 = nn.Conv3d(in_channel, out_channel,
			kernel_size=3, padding=1)
		self.conv_2 = nn.Conv3d(out_channel, out_channel,
			kernel_size=3, padding=1)

		pass
	
	def forward(self, x):

		out_1 = self.conv_1(x)
		out_2 = self.conv_2(F.leaky_relu(out_1))
		if self.residual:
			x = torch.cat([x] * int(self.concat_factor), dim=1)            
			out_3 = x + out_2
		else:
			out_3 = out_2
            
		return out_3


class multi_fusion(nn.Module):

	'''multi-resolution funsion unit
	'''

	def __init__(self, in_channel, out_channel):
		super(multi_fusion, self).__init__()

		self.conv_1 = nn.Conv3d(in_channel, out_channel,
			kernel_size=3, padding=1)
		self.conv_2 = nn.Conv3d(in_channel, out_channel,
			kernel_size=3, padding=1)
		self.conv_3 = nn.Conv3d(in_channel, out_channel,
			kernel_size=3, padding=1)
		
		pass

	def forward(self, x_1, x_2, x_3):

		out_1 = self.conv_1(x_1) # 256
		out_2 = self.conv_2(x_2) # 128
		out_3 = self.conv_3(x_3) # 64
		
		return out_1\
			+ F.upsample(out_2, scale_factor=2)\
			+ F.upsample(out_3, scale_factor=4)


class channel_residual_pooling(nn.Module):
	
	''' channel residual pooling unit
	'''

	def __init__(self, num_channel):
		super(channel_residual_pooling, self).__init__()

		self.pool_1 = nn.AvgPool3d(kernel_size=5, stride=1, padding=2)
		self.conv_1 = nn.Conv3d(num_channel, num_channel,
			kernel_size=3, padding=1)

		self.pool_2 = nn.AvgPool3d(kernel_size=5, stride=1, padding=2)
		self.conv_2 = nn.Conv3d(num_channel, num_channel,
			kernel_size=3, padding=1)

		pass

	def forward(self, x):

		out_1 = self.conv_1(self.pool_1(F.leaky_relu(x)))
		out_2 = self.conv_2(self.pool_2(out_1))

		return F.leaky_relu(x)\
			+ out_1\
			+ out_2


class refine_net(nn.Module):

	''' refine net 3d
	'''

	def __init__(self, num_classes, in_channels):
		super(refine_net, self).__init__()

		self.num_classes = num_classes

		self.conv_1 = residual_conv_unit(in_channels, 16) # for 256 resolution
		self.conv_2 = residual_conv_unit(in_channels, 16) # for 128 resolution
		self.conv_3 = residual_conv_unit(in_channels, 16) # for 64 resolution

		self.multi_fusion = multi_fusion(16, 16)
		self.channel_residual_pooling = channel_residual_pooling(16)

		self.out = residual_conv_unit(16, self.num_classes, residual=False)
		self.sigmoid = nn.Sigmoid()

		pass

	def forward(self, x_1, x_2, x_3):

		out_1 = self.conv_1(x_1)
		out_2 = self.conv_1(x_2)
		out_3 = self.conv_1(x_3)

		feat_map = self.multi_fusion(out_1, out_2, out_3)
		pool_map = self.channel_residual_pooling(feat_map)

		seg_map = self.sigmoid(self.out(pool_map))

		return seg_map

