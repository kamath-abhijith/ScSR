

# %% IMPORT LIBRARIES
import numpy as np
import pickle

from os import listdir, mkdir
from os.path import isdir

from skimage import io
from skimage.color import rgb2ycbcr, ycbcr2rgb, rgb2gray
from skimage.transform import resize

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

from ScSR import ScSR
from backprojection import backprojection

# %% PREPROCESSING FUNCTIONS

def normalize_signal(img, img_lr_ori, channel):
    if np.mean(img[:, :, channel]) * 255 > np.mean(img_lr_ori[:, :, channel]):
        ratio = np.mean(img_lr_ori[:, :, channel]) / (np.mean(img[:, :, channel]) * 255)
        img[:, :, channel] = np.multiply(img[:, :, channel], ratio)
    elif np.mean(img[:, :, channel]) * 255 < np.mean(img_lr_ori[:, :, channel]):
        ratio = np.mean(img_lr_ori[:, :, channel]) / (np.mean(img[:, :, channel]) * 255)
        img[:, :, channel] = np.multiply(img[:, :, channel], ratio)
    return img[:, :, channel]

def normalize_max(img):
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if img[m, n, 0] > 1:
                img[m, n, 0] = 1
            if img[m, n, 1] > 1:
                img[m, n, 1] = 1
            if img[m, n, 2] > 1:
                img[m, n, 2] = 1
    return img

# %% INITIALISE DICTIONARY

D_size = 1024
US_mag = 2
lmbd = 0.1
patch_size = 3

dict_name = str(D_size) + '_US' + str(US_mag) + '_L' + str(lmbd) + '_PS' + str(patch_size)

with open('data/dicts/Dh_' + dict_name + '.pkl', 'rb') as f:
    Dh = pickle.load(f)
Dh = normalize(Dh)
with open('data/dicts/Dl_' + dict_name + '.pkl', 'rb') as f:
    Dl = pickle.load(f)
Dl = normalize(Dl)

# %% INITIALISE PARAMETERS

# img_lr_dir = 'data/val_lr/'
# img_hr_dir = 'data/val_hr/'

img_lr_dir = 'data/set5_lr_sigma25/'
img_hr_dir = 'data/set5_hr/'

overlap = 1
lmbd = 0.3
upsample = 2
# colour_space = 'ycbcr'
colour_space = 'bw'
max_iteration = 1000

# %% VALIDATION

img_lr_file = listdir(img_lr_dir)

for i in tqdm(range(len(img_lr_file))):
# for i in tqdm(range(1)):
    # READ IMAGE AND MAKE FOLDERS
    img_name = img_lr_file[i]
    img_name_dir = list(img_name)
    img_name_dir = np.delete(np.delete(np.delete(np.delete(img_name_dir, -1), -1), -1), -1)
    img_name_dir = ''.join(img_name_dir)
    if isdir('data/results/set5_sigma25/' + dict_name + '_' + img_name_dir) == False:
        new_dir = mkdir('{}{}'.format('data/results/set5_sigma25/' + dict_name + '_', img_name_dir))
    img_lr = io.imread('{}{}'.format(img_lr_dir, img_name))

    ## READ AND SAVE ORIGINAL IMAGE
    img_hr = io.imread('{}{}'.format(img_hr_dir, img_name))
    io.imsave('{}{}{}{}'.format('data/results/set5_sigma25/' + dict_name + '_', img_name_dir, '/', '3HR.png'), img_hr)
    
    if colour_space == 'ycbcr':
        img_hr_y = rgb2ycbcr(img_hr)[:,:,0]

        ## CHANGE COLOUR SPACE
        img_lr_ori = img_lr
        temp = img_lr
        imr_lr = rgb2ycbcr(img_lr)
        img_lr_y = img_lr[:,:,0]
        img_lr_cb = img_lr[:,:,1]
        img_lr_cr = img_lr[:,:,2]

        ## UPSAMPLE CHROMINANCE CHANNEL DIRECTLY
        img_sr_cb = resize(img_lr_cb, (img_hr.shape[0], img_hr.shape[1]), order=0)
        img_sr_cr = resize(img_lr_cr, (img_hr.shape[0], img_hr.shape[1]), order=0)
    
    elif colour_space == 'bw':
        img_hr_y = rgb2gray(img_hr)
        img_lr = rgb2gray(img_lr)
        img_lr_y = img_lr
        img_lr_ori = img_lr

    ## SUPER-RESOLUTION OF LUMINANCE CHANNEL
    img_sr_y = ScSR(img_lr_y, img_hr_y.shape, upsample, Dh, Dl, lmbd, overlap)
    img_sr_y = backprojection(img_sr_y, img_lr_y, max_iteration)
    # img_sr_y = resize(img_lr_y, (img_hr.shape[0], img_hr.shape[1]), order=0) # Loop check

    ## RECONSTRUCT COLOUR IMAGE
    if colour_space == 'ycbcr':
        img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
        img_sr = ycbcr2rgb(img_sr)
        
        for channel in range(img_sr.shape[2]):
            img_sr[:, :, channel] = normalize_signal(img_sr, img_lr_ori, channel)

        img_sr = normalize_max(img_sr)
    
    elif colour_space == 'bw':
        img_sr = img_sr_y

    ## COMPUTE METRICS
    rmse_sr_hr = np.sqrt(mean_squared_error(img_hr_y, img_sr_y))
    # psnr_sr_hr = 10*np.log10(255.0**2/rmse_sr_hr**2)
    psnr_sr_hr = 10*np.log10(1.0**2/rmse_sr_hr**2)
    psnr_sr_hr = np.zeros((1,)) + psnr_sr_hr
    np.savetxt('{}{}{}{}'.format('data/results/set5_sigma25/' + dict_name + '_', img_name_dir, '/', 'PSNR_SR.txt'), psnr_sr_hr)

    ## SAVE SUPER-RESOLVED IMAGE
    io.imsave('{}{}{}{}'.format('data/results/set5_sigma25/' + dict_name + '_', img_name_dir, '/', '2SR.png'), img_sr)

# %% PLOTS

# fig, plts = plt.subplots(1,2,figsize=(10,6))
# plts[0].imshow(img_sr_y, cmap='gray')
# plts[0].set_title(r"Super-Resolved Lena PSNR: %.4f"%(psnr_sr_hr))

# plts[1].imshow(img_hr_y, cmap='gray')
# plts[1].set_title(r"Original Lena PSNR")

# # plt.show()

# %% PLOT DICTIONARY

fig, axs = plt.subplots(4,5, figsize=(15, 12), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(20):
    axs[i].imshow(Dl[:,i].reshape(6,6),cmap='gray')
    axs[i].axis('off')

plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Course Project/Slides/figures/dictionary.eps', format='eps')
plt.show()
# %%
