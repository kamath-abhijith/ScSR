
# %% IMPORT LIBRARIES
import numpy as np
import pickle

from os import listdir

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
from scipy.signal import convolve2d

from tqdm import tqdm

from spams import trainDL

# %% HELPER FUNCTIONS

def sample_patches(img, patch_size, patch_num, upscale):
    if img.shape[2] == 3:
        hIm = rgb2gray(img)
    else:
        hIm = img

    # Generate low resolution counter parts
    lIm = rescale(hIm, 1 / upscale)
    lIm = resize(lIm, hIm.shape)
    nrow, ncol = hIm.shape

    x = np.random.permutation(range(nrow - 2 * patch_size)) + patch_size
    y = np.random.permutation(range(ncol - 2 * patch_size)) + patch_size

    X, Y = np.meshgrid(x, y)
    xrow = np.ravel(X, order='F')
    ycol = np.ravel(Y, order='F')

    if patch_num < len(xrow):
        xrow = xrow[0 : patch_num]
        ycol = ycol[0 : patch_num]

    patch_num = len(xrow)

    H = np.zeros((patch_size ** 2, len(xrow)))
    L = np.zeros((4 * patch_size ** 2, len(xrow)))

    # Compute the first and second order gradients
    hf1 = [[-1, 0, 1], ] * 3
    vf1 = np.transpose(hf1)

    lImG11 = convolve2d(lIm, hf1, 'same')
    lImG12 = convolve2d(lIm, vf1, 'same')

    hf2 = [[1, 0, -2, 0, 1], ] * 3
    vf2 = np.transpose(hf2)

    lImG21 = convolve2d(lIm, hf2, 'same')
    lImG22 = convolve2d(lIm, vf2, 'same')

    for i in tqdm(range(patch_num)):
        row = xrow[i]
        col = ycol[i]

        Hpatch = np.ravel(hIm[row : row + patch_size, col : col + patch_size], order='F')
        # Hpatch = np.reshape(Hpatch, (Hpatch.shape[0], 1))
        
        Lpatch1 = np.ravel(lImG11[row : row + patch_size, col : col + patch_size], order='F')
        Lpatch1 = np.reshape(Lpatch1, (Lpatch1.shape[0], 1))
        Lpatch2 = np.ravel(lImG12[row : row + patch_size, col : col + patch_size], order='F')
        Lpatch2 = np.reshape(Lpatch2, (Lpatch2.shape[0], 1))
        Lpatch3 = np.ravel(lImG21[row : row + patch_size, col : col + patch_size], order='F')
        Lpatch3 = np.reshape(Lpatch3, (Lpatch3.shape[0], 1))
        Lpatch4 = np.ravel(lImG22[row : row + patch_size, col : col + patch_size], order='F')
        Lpatch4 = np.reshape(Lpatch4, (Lpatch4.shape[0], 1))

        Lpatch = np.concatenate((Lpatch1, Lpatch2, Lpatch3, Lpatch4), axis=1)
        Lpatch = np.ravel(Lpatch, order='F')

        if i == 0:
            HP = np.zeros((Hpatch.shape[0], 1))
            LP = np.zeros((Lpatch.shape[0], 1))
            # print(HP.shape)
            HP[:, i] = Hpatch - np.mean(Hpatch)
            LP[:, i] = Lpatch
        else:
            HP_temp = Hpatch - np.mean(Hpatch)
            HP_temp = np.reshape(HP_temp, (HP_temp.shape[0], 1))
            HP = np.concatenate((HP, HP_temp), axis=1)
            LP_temp = Lpatch
            LP_temp = np.reshape(LP_temp, (LP_temp.shape[0], 1))
            LP = np.concatenate((LP, LP_temp), axis=1)
    
    return HP, LP

def rnd_smp_patch(img_path, patch_size, num_patch, upsample):
    
    img_dir = listdir(img_path)

    img_num = len(img_dir)
    nper_img = np.zeros((img_num, 1))

    for i in tqdm(range(img_num)):
        img = io.imread('{}{}'.format(img_path, img_dir[i]))
        nper_img[i] = img.shape[0] * img.shape[1]

    nper_img = np.floor(nper_img * num_patch / np.sum(nper_img, axis=0))

    for i in tqdm(range(img_num)):
        patch_num = int(nper_img[i])
        img = io.imread('{}{}'.format(img_path, img_dir[i]))
        H, L = sample_patches(img, patch_size, patch_num, upsample)
        if i == 0:
            Xh = H
            Xl = L
        else:
            Xh = np.concatenate((Xh, H), axis=1)
            Xl = np.concatenate((Xl, L), axis=1)
    return Xh, Xl

def patch_pruning(Xh, Xl):
    pvars = np.var(Xh, axis=0)
    threshold = np.percentile(pvars, 10)
    idx = pvars > threshold
    # print(pvars)
    Xh = Xh[:, idx]
    Xl = Xl[:, idx]
    return Xh, Xl

# %% INITIALISE PARAMETERS

dict_size = 1024
lmbd = 0.1
patch_size = 3
num_samples = 1000000
upsample_factor = 2

# %% READ TRAINING IMAGES

train_img_path = 'data/train_hr/'

Xh, Xl = rnd_smp_patch(train_img_path, patch_size, num_samples, upsample_factor)

Xh, Xl = patch_pruning(Xh, Xl)
Xh = np.asfortranarray(Xh)
Xl = np.asfortranarray(Xl)

# %% DICTIONARY LEARNING
Dh = trainDL(Xh, K=dict_size, lambda1=lmbd, iter=100)
Dl = trainDL(Xl, K=dict_size, lambda1=lmbd, iter=100)

# %% SAVE DICTIONARIES
with open('data/dicts/'+ 'Dh_' + str(dict_size) + '_US' + str(upsample_factor) + '_L' + str(lmbd) + '_PS' + str(patch_size) + '.pkl', 'wb') as f:
    pickle.dump(Dh, f, pickle.HIGHEST_PROTOCOL)

with open('data/dicts/'+ 'Dl_' + str(dict_size) + '_US' + str(upsample_factor) + '_L' + str(lmbd) + '_PS' + str(patch_size) + '.pkl', 'wb') as f:
    pickle.dump(Dl, f, pickle.HIGHEST_PROTOCOL)
# %%
