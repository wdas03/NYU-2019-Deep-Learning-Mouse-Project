'''
Notes:
	* N class Vnet for single channel 3d images
Args:
	* X of [BatchSize, Channel = 1, X-dim, Y-dim, Z-dim]
Out:
	* Y of [BatchSize, NumClasses, X-dim, Y-dim, Z-dim]
Source: 
	* This vnet.py implement forked from https://github.com/mattmacy/vnet.pytorch
Bug fix: 
	* Broadcasting bug in class InputTransition
	* ContBatchNorm3d substituded with native
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def passthrough(x, **kwargs):
	return x

def ELUCons(elu, nchan):
	'''
	Notes: 
		* using leaky activation function!
	Args:
		* elu: bool, using ELU activation or Not
		* nchan: int, number of channel for PReLU
	Return:
		* nn.Module
	'''
	if elu==True:
		return nn.ELU(inplace=True)
	else:
		return nn.PReLU(nchan)

class LUConv(nn.Module):
	'''
	Notes: 
		* 3d conv->bn->RELU, retain channel number
		* nchan: number of channel in and out
	'''
	def __init__(self, nchan, elu):
		super(LUConv, self).__init__()
		self.relu1 = ELUCons(elu, nchan)
		self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
		self.bn1 = nn.BatchNorm3d(nchan)

	def forward(self, x):
		out = self.relu1(self.bn1(self.conv1(x)))
		return out

class ResLUConv(nn.Module):
	# bottle neck component
	def __init__(self, nchan, elu):
		super(ResLUConv, self).__init__()
		self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
		self.bn1 = nn.BatchNorm3d(nchan)
		self.relu1 = ELUCons(elu, nchan)

		self.conv2 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
		self.bn2 = nn.BatchNorm3d(nchan)
		self.relu2 = ELUCons(elu, nchan)

	def forward(self, x):

		out = self.relu1(self.bn1(self.conv1(x)))
		out = self.relu2(self.bn2(self.conv2(out)) + x)

		return out

def _make_rConv(nchan ,depth, elu):
	layers = []
	for _ in range(depth):
		layers.append(ResLUConv(nchan, elu))
	return nn.Sequential(*layers)

def _make_nConv(nchan, depth, elu):
	'''
	Notes:	
		* packaged conv3D layers
		* number = {2, 3}
	'''
	layers = []
	for _ in range(depth):
		layers.append(LUConv(nchan, elu))
	return nn.Sequential(*layers)

class InputTransition(nn.Module):
	'''
	Notes: 
		* X -> conv-> bn + X -> Relu
		* Bug fix in x16 = torch.cat
	'''
	def __init__(self, in_channels, outChans, elu):
		super(InputTransition, self).__init__()
		self.in_channels = in_channels
		self.conv1 = nn.Conv3d(in_channels, outChans, kernel_size=5, padding=2)
		self.bn1 = nn.BatchNorm3d(outChans)
		self.relu1 = ELUCons(elu, outChans)

	def forward(self, x):
		# do we want a PRELU here as well?
		out = self.bn1(self.conv1(x))
		x16 = torch.cat([x] * int(16 / self.in_channels), dim=1)
		out = self.relu1(torch.add(out, x16))

		return out

class DownTransition(nn.Module):
	'''
	Notes:
		* input -> conv/2-> bn -> relu -> X -> n*(conv3d->bn->relu) + X -> relu -> out
	'''
	def __init__(self, inChans, nConvs, elu, dropout=False, res=False):
		super(DownTransition, self).__init__()
		outChans = 2*inChans
		self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
		self.bn1 = nn.BatchNorm3d(outChans)
		self.do1 = passthrough
		self.relu1 = ELUCons(elu, outChans)
		self.relu2 = ELUCons(elu, outChans)
		if dropout:
			self.do1 = nn.Dropout3d()

		if res:
			self.ops = _make_rConv(outChans, nConvs, elu)
		else:
			self.ops = _make_nConv(outChans, nConvs, elu)

	def forward(self, x):
		down = self.relu1(self.bn1(self.down_conv(x)))
		out = self.do1(down)
		out = self.ops(out)
		out = self.relu2(torch.add(out, down))
		return out

class UpTransition(nn.Module):
	def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
		super(UpTransition, self).__init__()
		self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
		self.bn1 = nn.BatchNorm3d(outChans // 2)
		self.do1 = passthrough
		self.do2 = nn.Dropout3d()
		self.relu1 = ELUCons(elu, outChans // 2)
		self.relu2 = ELUCons(elu, outChans)
		if dropout:
			self.do1 = nn.Dropout3d()
		self.ops = _make_nConv(outChans, nConvs, elu)

	def forward(self, x, skipx):
		out = self.do1(x)
		skipxdo = self.do2(skipx)
		out = self.relu1(self.bn1(self.up_conv(out)))
		xcat = torch.cat((out, skipxdo), 1)
		out = self.ops(xcat)
		out = self.relu2(torch.add(out, xcat))
		return out

class OutputTransition(nn.Module):
	def __init__(self, inChans, classnum, elu):

		'''
		Notes: 
			* converts to number of outputs
		Args:
			* inChans: input channels
			* classnum: number of classes
		Return:
			* None
		'''
		super(OutputTransition, self).__init__()
		class_num = 3
		self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
		self.bn1 = nn.BatchNorm3d(2)
		self.conv2 = nn.Conv3d(2, classnum, kernel_size=1)
		self.relu1 = ELUCons(elu, 2)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# convolve 32 down to 2 channels
		out = self.relu1(self.bn1(self.conv1(x)))
		out = self.conv2(out)
		out = self.sigmoid(out)

		# out should have shape N, C, X, Y, Z at that time
		return out

class FCRB(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FCRB, self).__init__()
        self.affine = nn.Linear(in_chan, out_chan)
        self.bn = nn.BatchNorm1d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout()

    def forward(self, x):
        out = self.dp(self.relu(self.bn(self.affine(x))))
        return out

class LNet(nn.Module):
	'''
	Half V Net
	'''
	def __init__(self, img_size, out_size=6, elu=True):
		'''
		Args:
			* slim: using few conv layers, else as original paper
			* elu: using elu / PReLU
		'''
		super(LNet, self).__init__()

		x, y, z = img_size

		self.in_tr = InputTransition(16, elu)
		self.down_tr32 = DownTransition(16, 2, elu, dropout=True) # /2
		self.down_tr64 = DownTransition(32, 3, elu, dropout=True) # /4
		self.down_tr128 = DownTransition(64, 3, elu, dropout=True) # /8
		self.down_tr256 = DownTransition(128, 3, elu, dropout=True) # /16
		self.down_tr512 = DownTransition(256, 6, elu, dropout=True, res=True) # /32, 8 res fcn layers

		self.gap = nn.AvgPool3d(kernel_size = (x//32,y//32,z//32)) # N, C, 1, 1, 1

		channel_num = 512

		self.fc1 = FCRB(channel_num, channel_num)
		self.fc2 = nn.Linear(channel_num, out_size)

	def forward(self, x):
		batch_size = x.size(0) # get batch size

		out = self.in_tr(x)
		out = self.down_tr32(out)
		out = self.down_tr64(out)
		out = self.down_tr128(out)
		out = self.down_tr256(out)
		out = self.down_tr512(out)

		out = self.gap(out)
		out = out.view(batch_size, -1)

		out = self.fc1(out)
		out = self.fc2(out)
        
		return out

class LNetNew(nn.Module):
	'''
	Half V Net
	'''
	def __init__(self, img_size, out_size=3, elu=True):
		'''
		Args:
			* slim: using few conv layers, else as original paper
			* elu: using elu / PReLU
		'''
		super(LNetNew, self).__init__()

		x, y, z = img_size

		self.in_tr = InputTransition(8, elu)
		self.down_tr32 = DownTransition(8, 2, elu, dropout=True) # /2
		self.down_tr64 = DownTransition(16, 2, elu, dropout=True) # /4
		self.down_tr128 = DownTransition(32, 2, elu, dropout=True) # /8
		self.down_tr256 = DownTransition(64, 2, elu, dropout=True) # /16
		self.down_tr512 = DownTransition(128, 2, elu, dropout=True) # /32 = 4
		self.down_tr1024 = DownTransition(256, 3, elu, dropout=True, res=True) # /64 = 2

		channel_num = 512*2*2*2

		self.fc1 = FCRB(channel_num, out_size)

	def forward(self, x):

		batch_size = x.size(0) # get batch size

		out = self.in_tr(x)
		out = self.down_tr32(out)
		out = self.down_tr64(out)
		out = self.down_tr128(out)
		out = self.down_tr256(out)
		out = self.down_tr512(out)
		out = self.down_tr1024(out)

		out = out.view(batch_size, -1)

		out = self.fc1(out)
        
		return out


class DVNet(nn.Module):
	'''
	Deep V NET for 192 256 256 FULL FUCKING SIZE
	'''

	def __init__(self, classnum=1, elu=True):
		super(DVNet, self).__init__()

		self.in_tr = InputTransition(8, elu)

		self.down_tr16 = DownTransition(8, 2, elu, dropout=True)
		self.down_tr32 = DownTransition(16, 2, elu, dropout=True)
		self.down_tr64 = DownTransition(32, 3, elu, dropout=True)
		self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
		self.down_tr256 = DownTransition(128, 3, elu, dropout=True)

		self.up_tr256 = UpTransition(256, 256, 6, elu, dropout=True)
		self.up_tr128 = UpTransition(256, 128, 3, elu, dropout=True)
		self.up_tr64 = UpTransition(128, 64, 2, elu, dropout=True)
		self.up_tr32 = UpTransition(64, 32, 2, elu, dropout=True)
		self.up_tr16 = UpTransition(32, 16, 2, elu, dropout=True)

		self.out_tr = OutputTransition(16, classnum ,elu)

	def forward(self, x):

		out8 = self.in_tr(x)
		out16 = self.down_tr16(out8)
		out32 = self.down_tr32(out16)
		out64 = self.down_tr64(out32)
		out128 = self.down_tr128(out64)
		out256 = self.down_tr256(out128)
		out = self.up_tr256(out256, out128)
		out = self.up_tr128(out, out64)
		out = self.up_tr64(out, out32)
		out = self.up_tr32(out, out16)
		out = self.up_tr16(out, out8)
		out = self.out_tr(out)


class VNet(nn.Module):
	'''
	Note:
		VNet architecture As diagram of paper
	'''
	def __init__(self, classnum=1, in_channels=1, slim=True, elu=True):
		'''
		Args:
			* slim: using few conv layers, else as original paper
			* elu: using elu / PReLU
		'''
		super(VNet, self).__init__()

		self.slim=slim

		if slim:
			# 1 1 2 6 2 1 
			self.in_tr = InputTransition(in_channels, 16, elu)
			self.down_tr32 = DownTransition(16, 2, elu, dropout=True)
			self.down_tr64 = DownTransition(32, 2, elu, dropout=True)
			self.down_tr128 = DownTransition(64, 2, elu, dropout=True)
			self.down_tr256 = DownTransition(128, 3, elu, dropout=True)
			self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
			self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
			self.up_tr64 = UpTransition(128, 64, 2, elu, dropout=True)
			self.up_tr32 = UpTransition(64, 32, 2, elu, dropout=True)
			self.out_tr = OutputTransition(32, classnum ,elu)

		else:
			self.in_tr = InputTransition(in_channels, 16, elu)
			self.down_tr32 = DownTransition(16, 2, elu, dropout=True)
			self.down_tr64 = DownTransition(32, 3, elu, dropout=True)
			self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
			self.down_tr256 = DownTransition(128, 3, elu, dropout=True)
			self.up_tr256 = UpTransition(256, 256, 3, elu, dropout=True)
			self.up_tr128 = UpTransition(256, 128, 3, elu, dropout=True)
			self.up_tr64 = UpTransition(128, 64, 3, elu, dropout=True)
			self.up_tr32 = UpTransition(64, 32, 2, elu, dropout=True)
			self.out_tr = OutputTransition(32, classnum ,elu)
			

	def forward(self, x):

		if self.slim:
			out16 = self.in_tr(x)
			out32 = self.down_tr32(out16)
			out64 = self.down_tr64(out32)
			out128 = self.down_tr128(out64)
			out256 = self.down_tr256(out128)
			out = self.up_tr256(out256, out128)
			out = self.up_tr128(out, out64)
			out = self.up_tr64(out, out32)
			out = self.up_tr32(out, out16)
			out = self.out_tr(out)

		else:
			pass
			
			out16 = self.in_tr(x)
			out32 = self.down_tr32(out16)
			out64 = self.down_tr64(out32)
			out128 = self.down_tr128(out64)
			out256 = self.down_tr256(out128)
			out = self.up_tr256(out256, out128)
			out = self.up_tr128(out, out64)
			out = self.up_tr64(out, out32)
			out = self.up_tr32(out, out16)
			out = self.out_tr(out)
			
		return out

class WNet(nn.Module):

	def __init__(self):

		super(WNet, self).__init__()
		self.VNet1 = VNet()
		self.VNet2 = VNet()

	def forward(self, x):
		body = self.VNet1(x)

		bodyMask = (body-body.min())/(body.max()-body.min())
		bodyMask = torch.round(bodyMask)

		bv = self.VNet2(x*bodyMask)

		return torch.cat((body, bv), 1)

class VNetMask(nn.Module):
	'''
	Note:
		VNet architecture As diagram of paper
	'''
	def __init__(self, elu=True):
		'''
		Args:
			* slim: using few conv layers, else as original paper
			* elu: using elu / PReLU
		'''
		super(VNetMask, self).__init__()

		self.in_tr = InputTransition(16, elu)
		self.down_tr32 = DownTransition(16, 1, elu)
		self.down_tr64 = DownTransition(32, 1, elu)
		self.down_tr128 = DownTransition(64, 2, elu, dropout=True)
		self.up_tr128 = UpTransition(128, 128, 8, elu, dropout=True)
		self.up_tr64 = UpTransition(128, 64, 2, elu)
		self.up_tr32 = UpTransition(64, 32, 1, elu)
		self.out_tr = OutputTransition(32, 3, elu) # BKG, Body Segmentation map

	def forward(self, x):

		out16 = self.in_tr(x)
		out32 = self.down_tr32(out16)
		out64 = self.down_tr64(out32)
		out128 = self.down_tr128(out64)
		out = self.up_tr128(out128, out64)
		out = self.up_tr64(out, out32)
		outfeature = self.up_tr32(out, out16)
		outmap = self.out_tr(outfeature) # this is the the Body Segmentation map

		outbkg = outmap.narrow(1, 0, 1)
		outbody = outmap.narrow(1, 1, 1)
		outbv = outmap.narrow(1, 2, 1)

		bodymsk = (outbody-outbody.min())/(outbody.max()-outbody.min()) # remap the mask to [0, 1]

		outbv = outbv*bodymsk 

		return torch.cat((outbkg, outbody, outbv), 1)

