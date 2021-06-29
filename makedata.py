## SET THE SCALE PARAMETER
## SET THE INPUT AND OUTPUT IMAGE PATHS

# %% IMPORT LIBRARIES
import numpy as np

from os import listdir
from tqdm import tqdm

from skimage import io
from skimage.transform import rescale

# %% SET PATHS
train_hr_path = 'data/train_hr/'
train_lr_path = 'data/train_lr/'

# val_hr_path = 'data/val_hr/'
# val_lr_path = 'data/val_lr/'

val_hr_path = 'data/set5_hr/'
val_lr_path = 'data/set5_lr/'

scale = 2

# %% DOWNSAMPLE TRAINING IMAGES

num_train_images = len(listdir(train_hr_path))
for i in tqdm(range(num_train_images)):
    img_name = listdir(train_hr_path)[i]
    img = io.imread('{}{}'.format(train_hr_path, img_name))

    scaled_image = rescale(img, (1/scale), multichannel=True, anti_aliasing=1)
    io.imsave('{}{}'.format(train_lr_path, img_name), scaled_image)

# %% DOWNSAMPLE VALIDATION IMAGES

num_val_images = len(listdir(val_hr_path))
for i in tqdm(range(num_val_images)):
    img_name = listdir(val_hr_path)[i]
    img = io.imread('{}{}'.format(val_hr_path, img_name))

    scaled_image = rescale(img, (1/scale), multichannel=True, anti_aliasing=1)
    io.imsave('{}{}'.format(val_lr_path, img_name), scaled_image)

# %% DOWNSAMPLE + ADD NOISE ON VALIDATION IMAGES

# sigma = np.array([0,4,8,16,25])
sigma = 0/255

num_val_images = len(listdir(val_hr_path))
for i in tqdm(range(num_val_images)):
    img_name = listdir(val_hr_path)[i]
    img = io.imread('{}{}'.format(val_hr_path, img_name))

    scaled_image = rescale(img, (1/scale), multichannel=True, anti_aliasing=1)
    noisy_image = scaled_image + sigma*np.random.randn(scaled_image.shape[0], scaled_image.shape[1], scaled_image.shape[2])
    io.imsave('{}{}'.format(val_lr_path, img_name), noisy_image)

# %%
