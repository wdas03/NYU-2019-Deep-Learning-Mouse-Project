import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def dice_loss(input, target):

	''' standard soft dice coefficience

	Parameters
	----------
	input: tensor, predicted segmentation map with size = None, 1, X*3, [0,1]
	target: tensor, truth segmentation map with size = None, 1, X*3, {0,1}

	Returns
	-------
	loss: 1 - dice coefficient
	'''

	assert input.size() == target.size()
	eplison = 1e-6

	dice_coefficient = torch.sum(2*input*target, dim=(2,3,4))/ \
			(torch.sum(input*input,dim=(2,3,4))+torch.sum(target*target,dim=(2,3,4))+eplison)

	# Average dice coeff over mini batch

	return 1 - torch.mean(dice_coefficient)


def dice_loss_3(input, target):

	''' standard soft dice coefficience for 3 classes

	Parameters
	----------
	input: tensor, predicted segmentation map with size = None, 3, X*3, [0,1]
	target: tensor, truth segmentation map with size = None, 3, X*3, {0,1}

	Returns
	-------
	loss: scalar tensor, average loss over classes
	'''

	loss = ((dice_loss(input.narrow(1,0,1), target.narrow(1,0,1))) \
			+ (dice_loss(input.narrow(1,1,1), target.narrow(1,1,1))) \
			+ (dice_loss(input.narrow(1,2,1), target.narrow(1,2,1))))/3

	return loss

def dice_loss_3_debug(input, target):
	loss_1 = (dice_loss(input.narrow(1,0,1), target.narrow(1,0,1)))
	loss_2 = (dice_loss(input.narrow(1,1,1), target.narrow(1,1,1)))
	loss_3 = (dice_loss(input.narrow(1,2,1), target.narrow(1,2,1)))

	return loss_1, loss_2, loss_3

