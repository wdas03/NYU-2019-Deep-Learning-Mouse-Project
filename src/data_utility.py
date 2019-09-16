import os

import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from scipy.ndimage import affine_transform
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import imgaug.augmenters as iaa

def get_min_validation(filename):
    file = open(filename, "r")
    min_val = 1
    
    lines = file.readlines()
    epoch = 0
    
    for idx, line in enumerate(lines):
        if line.startswith("-"):
            val = float(line[-7:].strip())
            if val < min_val:
                min_val = val
                previous_line = lines[idx - 1].strip()
                #print(previous_line)
                epoch = int(previous_line.split(' ')[1])
    
    return min_val, epoch

def find_file_with_epoch(epoch_num, folder):
    out = ''
    for file in sorted(os.listdir(folder)):
        filename = os.path.join(folder, file)
        if int(os.path.splitext(filename.split(' ')[-1])[0]) == epoch_num:
            out = filename
    return out


def generate_pyramid(image):
	
	''' do data normalization, generating a 3 level image pyramid

	Parameters
	----------
	image: ndarry, shape = (3,256*3)

	Returns
	image1, image2, image4: ndarray, shape = (3, 256*3), (3, 128*3), (3, 64*3)
	'''

	image_mean = 50.4328 # measured mean
	image_std = 71.6100 # measured std

	image1 = image
	image1[0] = (image[0] - image_mean)/image_std

	channel, xshape, yshape, zshape = image.shape
	image2 = np.zeros([channel, xshape//2, yshape//2, zshape//2], dtype=np.float32)
	image4 = np.zeros([channel, xshape//4, yshape//4, zshape//4], dtype=np.float32)

	for c in np.arange(channel):

		if c == 0:
			# order = 3
			image2[c] = zoom(image1[c], zoom=0.5, order=3)
			image4[c] = zoom(image1[c], zoom=0.25, order=3) 

		else:
			# order = 0
			image2[c] = zoom(image1[c], zoom=0.5, order=0)
			image4[c] = zoom(image1[c], zoom=0.25, order=0) 

	return image1, image2, image4


def reshape_image (image, target_x, target_y, target_z, order=3):

	"""	reshaping the img to desirable shape through zero pad or 
	zoom according to longest axis, some of the code might be 
	redundent

	Parameters
	----------
	image: input 3d nii array
	target_x: target shape x
	target_y: target shape y
	target_z: target shape z
	
	Returns
	-------
	image: reshaped 3d nii array

	"""
	if torch.is_tensor(image):
		image = image.cpu().detach().numpy()
        
	maxdimension = np.max(image.shape)
	maxtarget = np.max([target_x, target_y, target_z])

	if maxdimension > maxtarget:
		image = zoom(image, zoom=maxtarget/maxdimension, order=order)
		# print('zoomed!', image.shape)

	padx = (target_x-image.shape[0])//2
	pady = (target_y-image.shape[1])//2
	padz = (target_z-image.shape[2])//2

	# some magic code that successfully distinguish odd and even bug
	extrax = int(image.shape[0]%2!=0)
	extray = int(image.shape[1]%2!=0)
	extraz = int(image.shape[2]%2!=0)

	zero_array = ((0,0),(0,0),(0,0))

	image = np.pad (image, ((extrax + padx,padx),(0, 0),(0, 0)), \
		mode='constant', constant_values=zero_array)

	image = np.pad (image, ((0,0),(extray + pady, pady),(0, 0)), \
			mode='constant', constant_values=zero_array)

	image = np.pad (image, ((0,0),(0, 0),(extraz + padz, padz)), \
			mode='constant', constant_values=zero_array)

	#print('the image shape is = {}'.format(image.shape))

	return torch.from_numpy(image)


def show_image_slices(image):
    if torch.is_tensor(image):
        image = image.cpu().detach().numpy()
    
    image = np.squeeze(image)
    print(image.shape)
    if (len(image.shape)) == 3:
        x,y,z = image.shape
        plt.imshow(image[x//2], cmap='gray')
    plt.show()
    pass

def show_image_slice(image):

	''' show one slice of image, by default from the middle

	Parameters
	----------
	image: input 3d nii array, either 3d, 4d or 5d with first dimension as 1

	Returns
	-------
	None

	'''

	if torch.is_tensor(image):
		image = image.cpu().detach().numpy()

	image = np.squeeze(image)

	if (len(image.shape)) == 3:
		# x, y, z
		x,y,z = image.shape
		plt.imshow(image[x//2], cmap='gray')

	elif (len(image.shape)) == 4:
		# c, x, y, z
		channel,x,y,z = image.shape
		fig, ax = plt.subplots(1, channel, sharey=True)

		for c in np.arange(channel):
			ax[c].imshow(image[c][x//2],  cmap='gray')

	else:
		# 1, c, x, y, z
		_,channel,x,y,z = image.shape
		fig, ax = plt.subplots(1, channel, sharey=True)

		for c in np.arange(channel):
			ax[c].imshow(image[0][c][x//2],  cmap='gray')

	plt.show()

	pass

'''
def multi_slice_viewer(volume):
    volume = np.squeeze(volume)
    #remove_keymap_conflicts({'w', 'u'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'w':
        previous_slice(ax)
    elif event.key == 'u':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
'''

def get_path(data_type, index):

	''' get the files path of an image of certain type

	Parameters
	----------
	data_type: str, {'nii_hard', 'nii_test', 'nii_train'}
	index: int, image index, train: [0, 106]

	Returns
	-------
	[]: list of image_path, body_path, bv_path: 

	'''

	counter = 0
	paths = []
	for root, dirs, files in os.walk('../data/' + data_type, topdown=False):
		
		# need to declear this explicitly for linux, see url
		# https://stackoverflow.com/questions/18282370/in-what-order-does-os-walk-iterates-iterate
		
		files.sort()
		for name in files:
			if counter//3 == index:
				paths.append(os.path.join(root, name))
			counter += 1
	return paths

def show_label(label):
    show_image_slice(label)

def get_image(data_type, index, verbose=False):

	''' get the image file of certain type

	Parameters
	----------
	data_type: str, {'nii_hard', 'nii_test', 'nii_train'}
	index: int, image index, train: [0, 106]

	Returns
	-------
	image: ndarray with shape = {2, 256, 256, 256}, {image, body_label + bv_label}

	0: background
	1: body
	2: bv

	'''
	image_path, body_path, bv_path = get_path(data_type, index)

	image = ((nib.load(image_path)).get_fdata()).astype(np.float32)
	body = ((nib.load(body_path)).get_fdata()).astype(np.float32)
	bv = ((nib.load(bv_path)).get_fdata()).astype(np.float32)

	image_reshaped = reshape_image(image, 256, 256, 256)
	body_reshaped = reshape_image(body, 256, 256, 256, order=0)
	bv_reshaped = reshape_image(bv, 256, 256, 256, order=0)

	image_reshaped = image_reshaped
    
	original_image_dimensions = image.shape
    
	if verbose:
		# for debugging purpose
		# print('before reshaping, image shape = {}, body shape = {}, bv shape = {}'.format(image.shape, body.shape, bv.shape))
		# print('after reshaping, image shape = {}, body shape = {}, bv shape = {}'.format(image_reshaped.shape, body_reshaped.shape, bv_reshaped.shape))
		show_image_slice(np.concatenate([image_reshaped[None], body_reshaped[None] + bv_reshaped[None]]))
        
	return np.concatenate([image_reshaped[None], body_reshaped[None] + bv_reshaped[None]]), original_image_dimensions

def filp_function(image, x, y, z):

	''' filp image

	Parameters
	----------
	image: ndarry, shape = {3, 256*3}
	x,y,z: bool, whether filp or not

	Returns
	-------
	image: ndarry, shape = {3, 256*3}

	'''
	bool_array = np.array([0,x,y,z], dtype=np.float32)
	axis_array = bool_array.nonzero()[0]

	image = np.flip(image, axis=axis_array)

	return image

class CropAndPad(object):
    def __init__(self):
        pass
    def __call__(self, image):
        seq = iaa.Sequential(
            iaa.CropAndPad(percent=(-0.25, 0.25))
        )
        aug = seq.augment_images(image)
        return aug

class ElasticTransformation(object):
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma
    def __call__(self, image):
        seq = iaa.Sequential([
            iaa.ElasticTransformation(alpha=self.alpha, sigma=self.sigma)
        ])
        aug = seq.augment_images(image)
        return aug

class random_filp(object):

	''' random filp callable object
	'''

	def __init__(self, p):

		''' randomly apply filp transform

		Parameters
		----------
		p: float, probablity that it filps

		Returns
		-------
		None

		'''

		self.p = p
		pass

	def __call__(self, image):

		''' override callable method, apply filp function
		'''

		x, y, z = np.random.uniform(0, 1, size=3)
		p = self.p

		image = filp_function(image, (x<p), (y<p), (z<p))

		return image


def affine_function(image, xr, yr, zr, xm, ym, zm):

	''' Rotate and move, MoveToCenter->RotateX->RotateY->RotateZ->MoveBack->MoveRandom
	
	Parameters
	----------
	img: image of shape (C=3, X, Y, Z)
	xr, yr, zr: Rotate in degree
	xm, ym, zm: move as int

	Returns
	-------
	image: Transformed image of shape (C=3, X, Y, Z)
	'''

	sinx = np.sin(np.deg2rad(xr))
	cosx = np.cos(np.deg2rad(xr))

	siny = np.sin(np.deg2rad(yr))
	cosy = np.cos(np.deg2rad(yr))

	sinz = np.sin(np.deg2rad(zr))
	cosz = np.cos(np.deg2rad(zr))

	xc = image[0].shape[0]//2
	yc = image[0].shape[1]//2
	zc = image[0].shape[2]//2

	Mc = np.array([[1, 0, 0, xc],
				[0, 1, 0, yc],
				[0, 0, 1, zc],
				[0, 0, 0, 1]])

	Rx = np.array([[cosx, sinx, 0, 0],
				[-sinx, cosx, 0, 0],
				[0, 0, 1, 0], 
				[0, 0, 0, 1]])

	Ry = np.array([[cosy, 0, siny, 0],
				[0, 1, 0, 0],
				[-siny, 0, cosy, 0], 
				[0, 0, 0, 1]])

	Rz = np.array([[1, 0, 0, 0],
				[0, cosz, sinz, 0],
				[0, -sinz, cosz, 0], 
				[0, 0, 0, 1]])
	
	Mb = np.array([[1, 0, 0, -xc],
				[0, 1, 0, -yc],
				[0, 0, 1, -zc],
				[0, 0, 0, 1]])
	
	MM = np.array([[1, 0, 0, xm],
				[0, 1, 0, ym],
				[0, 0, 1, zm],
				[0 ,0, 0, 1]])

	Matrix = np.linalg.multi_dot([Mc, Rx, Ry, Rz, Mb, MM])

	channel,_,_,_ = image.shape
	
	for c in np.arange(channel):
		if c == 0:
			# image, use order 3
			image[c] = affine_transform(image[c], Matrix, output_shape=image[c].shape, order=3)
		else:
			# label, use order 0
			image[c] = affine_transform(image[c], Matrix, output_shape=image[c].shape, order=0)

	return image


class random_affine(object):

	''' random affine callable object
	'''

	def __init__(self, fluR, fluM):

		''' randomly apply affine transform

		Parameters
		----------
		fluR: float, flunctuation in rotation
		fluM: float, flunctuation in move

		Returns
		-------
		image: ndarry, shape = {3, 256*3}

		'''

		self.fluR = fluR
		self.fluM = fluM

	def __call__(self, image):

		''' override callable method, apply affine_function
		'''

		xr, yr, zr = np.random.uniform(-self.fluR, self.fluR, size=3)
		xm, ym, zm = np.random.uniform(-self.fluM, self.fluM, size=3)

		image = affine_function(image, xr, yr, zr, xm, ym, zm)

		return image
    
def toTensor(imagek):

	''' transform ndarray to tensor

	Parameters
	----------
	imagek: ndarry shape = (3, k*3)

	Returns
	-------
	imagek_data_tensor
	imagek_label_tensor

	'''

	imagek_data = imagek[0][None] # reshape to (1, 256*3)

	imagek_background = (imagek[1][None] == 0).astype(np.float32)
	imagek_body = (imagek[1][None] == 1).astype(np.float32)
	imagek_bv = (imagek[1][None] == 2).astype(np.float32)

	imagek_label = np.concatenate([imagek_background, imagek_body,imagek_bv]) # reshape to (2, 256*3)

	imagek_data_tensor = torch.from_numpy(imagek_data.copy())
	imagek_label_tensor = torch.from_numpy(imagek_label.copy())

	return imagek_data_tensor, imagek_label_tensor

class get_full_resolution_dataset(Dataset):
	def __init__(self, data_type, transform=None):
		
		''' Override init method

		Parameters
		----------
		data_type: str, {'nii_train', 'nii_test'}
		transform: [] of pytorch transform

		Returns
		-------
		None

		'''
		assert (data_type == 'nii_train' or data_type == 'nii_test')

		self.data_type = data_type
		self.transform = transform

		pass

	def __len__(self):

		''' Override return size of dataset

		Parameters
		----------
		None

		Returns
		-------
		int, 138 train or 46 test

		'''

		if self.data_type == 'nii_train':
			# having 138 train image
			return 138
		else:
			# and 46 test image
			return 46

		pass

	def __getitem__(self, index):

		''' Override return size of dataset

		Parameters
		----------
		index: int within 138 or within 46

		Returns
		-------
		image_dict: torch tensor with key = {}
		'''

		image, original_image_dimensions = get_image(self.data_type, index)
		#print(original_image_dimensions)
        
		if self.transform:
			image = self.transform(image)
			
		image1, image2, image4 = generate_pyramid(image)

		image1_data, image1_label = toTensor(image1)

		sample = {'image1_data': image1_data,
				'image1_label': image1_label,
				'original_resolution': original_image_dimensions
                 }

		return sample

class pyramid_dataset(Dataset):
	
	''' pytorch dataset object that spite out a dictionary
	'''

	def __init__(self, data_type, transform=None):
		
		''' Override init method

		Parameters
		----------
		data_type: str, {'nii_train', 'nii_test'}
		transform: [] of pytorch transform

		Returns
		-------
		None

		'''
		assert (data_type == 'nii_train' or data_type == 'nii_test')

		self.data_type = data_type
		self.transform = transform

		pass

	def __len__(self):

		''' Override return size of dataset

		Parameters
		----------
		None

		Returns
		-------
		int, 138 train or 46 test

		'''

		if self.data_type == 'nii_train':
			# having 138 train image
			return 138
		else:
			# and 46 test image
			return 46

		pass

	def __getitem__(self, index):

		''' Override return size of dataset

		Parameters
		----------
		index: int within 138 or within 46

		Returns
		-------
		image_dict: torch tensor with key = {}
		'''

		image, original_image_dimensions = get_image(self.data_type, index)
        
		#print(original_image_dimensions)
        
		if self.transform:
			image = self.transform(image)
			
		image1, image2, image4 = generate_pyramid(image)

		image1_data, image1_label = toTensor(image1)
		image2_data, image2_label = toTensor(image2)
		image4_data, image4_label = toTensor(image4)

		sample = {'image1_data': image1_data,
				'image1_label': image1_label,
				'image2_data': image2_data,
				'image2_label': image2_label,
				'image4_data': image4_data,
				'image4_label':image4_label}

		return sample