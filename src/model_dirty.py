import torch
import torch.nn as nn
import torch.nn.functional as F

''' Stacking V net to save memory
	Variance of image cascade network
	by Tongda Xu
'''

class output_unit(nn.Module):

	''' input unit with several conv layers

	Parameters
	----------
	out_channel: int, out channel 

	Callable
	--------
	return out, tensor, outchannel, x/2, 

	'''

	def __init__(self, input_channel, cat_map=False):
		super(output_unit, self).__init__()
		self.cat_map = cat_map

		self.conv1 = nn.Conv3d(input_channel,3,kernel_size=1) # linear combo
		# taking a linear combination, no bn nor relu follow

		self.sigmoid = nn.Softmax(dim=1)
		# taking softmax over channels

		pass

	def forward(self, x, map=None):

		if self.cat_map:
			conv_1 = self.conv1(torch.cat((x, map), dim=1))
		else:
			conv_1 = self.conv1(x)

		out = self.sigmoid(conv_1)

		return out


class input_unit(nn.Module):

	''' input unit with several conv layers

	Parameters
	----------
	out_channel: int, out channel 

	Callable
	--------
	return out, tensor, outchannel, x/2, 

	'''

	def __init__(self, out_channel):
		super(input_unit, self).__init__()

		self.conv1 = nn.Conv3d(1,out_channel,kernel_size=5,padding=2)
		self.bn1 = nn.BatchNorm3d(out_channel)
		
		pass

	def forward(self, x):

		conv_1 = self.bn1(self.conv1(x))
		out = F.leaky_relu(conv_1 + x) 
		# residual connection on the first layer or not?

		return out


class down_unit(nn.Module):

	''' down unit with down transition, 2 convolution and res connection

	Parameters
	----------
	in_channel: int, the input conv channel number
	out_channel: int, the output conv channel number 

	Callable
	--------
	return out, tensor, outchannel, x/2, 

	'''

	def __init__(self, in_channel, out_channel, add_feat_map=False):
		super(down_unit, self).__init__()

		self.add_feat_map = add_feat_map

		self.conv_down = nn.Conv3d(in_channel, out_channel, 
			kernel_size=2, stride=2)
		self.bn_down = nn.BatchNorm3d(out_channel)

		self.conv1 = nn.Conv3d(out_channel,out_channel,kernel_size=5,padding=2)
		self.bn1 = nn.BatchNorm3d(out_channel)

		self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size=5,padding=2)
		self.bn2 = nn.BatchNorm3d(out_channel)

		pass

	def forward(self, x, feat_map=None):

		down = F.leaky_relu((self.bn_down(self.conv_down(x))))
		
		if self.add_feat_map:
			# cat the feature map before convolution
			down = down + feat_map

		conv_1 = F.leaky_relu(self.bn1(self.conv1(down)))
		conv_2 = self.bn2(self.conv2(conv_1))
		out = F.leaky_relu(down + conv_2) # residual connection

		return out


class up_unit(nn.Module):

	''' down unit with down transition, 2 convolution and res connection
	a lean up unit to save memory
	
	Parameters
	----------
	in_channel: int, the input conv channel number
	out_channel: int, the output conv channel number 

	Callable
	--------
	return out, tensor, outchannel, x/2, 

	'''

	def __init__(self, in_channel, out_channel):
		super(up_unit, self).__init__()

		self.conv_up = nn.ConvTranspose3d(in_channel, out_channel, 
			kernel_size=2, stride=2)
		self.bn_up = nn.BatchNorm3d(out_channel)

		self.conv1 = nn.Conv3d(out_channel, out_channel, 
		 	kernel_size=5, padding=2)
		self.bn1 = nn.BatchNorm3d(out_channel)

		pass

	def forward(self, x):

		up = F.leaky_relu((self.bn_up(self.conv_up(x))))
		conv_1 = self.bn1(self.conv1(up))
		out = F.leaky_relu(up + conv_1) # residual connection

		return out


class net_4(nn.Module):

	def __init__(self, training=True):
		
		super(net_4, self).__init__()
		'''
		train: bool, if true, also train the decoder
		'''
		self.iu = input_unit(8) # out = 1/4 = 64, 8
		self.du_1 = down_unit(8, 16) # out = 1/8 = 32, 16
		self.du_2 = down_unit(16, 32) # out = 1/16 = 16, 32
		self.du_3 = down_unit(32, 64) # out = 1/32 = 8, 64
		self.istraining = training
		if self.training:
			self.uu_3 = up_unit(64, 32)
			self.uu_2 = up_unit(32, 16)
			self.uu_1 = up_unit(16, 8)
			self.ou = output_unit(8)

		pass

	def forward(self, x):

		out = self.iu(x)
		out = self.du_1(out)
		out = self.du_2(out)
		out = self.du_3(out)

		feat_map = out # 1/32, 64

		if self.istraining:

			out = self.uu_3(out)
			out = self.uu_2(out)
			out = self.uu_1(out)
			out = self.ou(out)

			return feat_map, out

		else:

			return feat_map


class net_2(nn.Module):

	def __init__(self, training=True):
		
		super(net_2, self).__init__()

		self.iu = input_unit(8) # out = 1/2 = 128, 8
		self.du_1 = down_unit(8, 16) # out = 1/4 = 64, 16
		self.du_2 = down_unit(16, 32, add_feat_map=True) # out = 1/8, 32

		self.up_feat = nn.ConvTranspose3d(64, 
				32, kernel_size=4, stride=4)
		self.istraining = training

		if self.training:

			self.uu_2 = up_unit(32, 16)
			self.uu_1 = up_unit(16, 8)
			self.ou = output_unit(8)

		pass

	def forward(self, x, in_map):

		out = self.iu(x)
		out = self.du_1(out)

		in_map_up = self.up_feat(in_map) # in_map_out = 1/8, 32
		out = self.du_2(out, in_map_up)

		feat_map = out

		if self.istraining:

			out = self.uu_2(out)
			out = self.uu_1(out)
			out = self.ou(out)

			return feat_map, out

		else:

			return feat_map


class net_1(nn.Module):

	def __init__(self):
		
		super(net_1, self).__init__()

		self.iu = input_unit(8) # out = 1/1 = 256, 8
		self.du_1 = down_unit(8, 16, add_feat_map=True) # out = 1/8, 16

		self.up_feat = nn.ConvTranspose3d(32, 
				16, kernel_size=4, stride=4)

		self.uu_1 = up_unit(16, 8) # out = 1/2 = 256, 8
		self.ou = output_unit(8*2, cat_map = True) # with concat of map

		pass

	def forward(self, x, in_map):

		out_in = self.iu(x)

		in_map_up = self.up_feat(in_map) # in_map_out = 1/8, 32
		out = self.du_1(out_in, in_map_up)

		out = self.uu_1(out)
		out = self.ou(out, out_in)

		return out

''' model version 2
'''

class net_4_v2(nn.Module):

	def __init__(self):
		
		super(net_4_v2, self).__init__()
		'''
		train: bool, if true, also train the decoder
		'''
		self.iu = input_unit(8) # out = 1/4 = 64, 8
		self.du_1 = down_unit(8, 16) # out = 1/8 = 32, 16
		self.du_2 = down_unit(16, 32) # out = 1/16 = 16, 32

		self.ou = self.ou = output_unit(32)

		pass

	def forward(self, x):

		out = self.iu(x)
		out = self.du_1(out)
		feat_map_16 = self.du_2(out) # 16^3 x 32 channels
		
		out_64 = F.upsample(feat_map_16, scale_factor=4)
		seg_64 = self.ou(out_64)

		return seg_64, feat_map_16 


class net_2_v2(nn.Module):

	def __init__(self):
		
		super(net_2_v2, self).__init__()
		'''
		train: bool, if true, also train the decoder
		'''
		self.iu = input_unit(8) # out = 1/2 = 128, 8
		self.du_1 = down_unit(8, 16) # out = 1/4 = 64, 16
		self.du_2 = down_unit(16, 32, add_feat_map=True) # out = 1/8 = 32, 32

		self.ou = self.ou = output_unit(32)

		pass

	def forward(self, x, feat_map_16):

		out = self.iu(x)
		out = self.du_1(out)
		feat_map_32 = self.du_2(out, F.upsample(feat_map_16, scale_factor=2)) # 16^3 x 32 channels
		
		out_128 = F.upsample(feat_map_32, scale_factor=4)
		seg_128 = self.ou(out_128)

		return seg_128, feat_map_32 


class net_1_v2(nn.Module):

	def __init__(self):
		
		super(net_1_v2, self).__init__()
		'''
		train: bool, if true, also train the decoder
		'''
		self.iu = input_unit(8) # out = 1/1 = 256, 8
		self.du_1 = down_unit(8, 16) # out = 1/2 = 128, 16
		self.du_2 = down_unit(16, 32, add_feat_map=True) # out = 1/4 = 64, 32

		self.uu_2 = up_unit(32, 16) # out = 1/2 = 128, 16*2 with cat from du_1
		self.uu_1 = up_unit(32, 8) # out = 1/1 = 256, 8*2 with cat from du_2
		self.ou = output_unit(16) # with concat of map

		pass

	def forward(self, x, feat_map_32):

		map_256 = self.iu(x) # 256^3 x 8
		map_128 = self.du_1(map_256) # 128^3 x 16
		map_64 = self.du_2(map_128, F.upsample(feat_map_32, scale_factor=2)) # 64^3 x 32 channels
		
		out_128 = self.uu_2(map_64)
		out_256 = self.uu_1(torch.cat((out_128, map_128), dim=1))
		seg_256 = self.ou(torch.cat((out_256, map_256), dim=1))

		return seg_256 


class net_v2(nn.Module):
	''' Image cascad network version 2
	'''
	def __init__(self):	
		super(net_v2, self).__init__()

		self.net_4 = net_4_v2()
		self.net_2 = net_2_v2()
		self.net_1 = net_1_v2()

	def forward(self, image_4, image_2, image_1):

		out_4, map_4  = model_4(image_4)
		out_2, map_2 = model_2(image_2, map_4)
		out_1 = model_1(image_1, map_2)

		return out_4, out_2, out_1

