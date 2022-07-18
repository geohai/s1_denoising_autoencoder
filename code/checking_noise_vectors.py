import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from random import randrange

from sentinel1denoised.s1denoise import Sentinel1Image
from torch.utils.data import Dataset


img_list = [f for f in os.listdir("../data/train/zip/") if f[-3:]=="zip"]

for img in img_list:
    s1 = Sentinel1Image("../data/train/zip/"+img)

    sigma0_hh = s1[s1.get_band_number(band_id="sigma0_HH")]
    sigma0_hh = sigma0_hh[100:-100, 100:-100]
    sigma0_hh = (np.log10(sigma0_hh) + 3.6) / 5.5
    # print(f"{np.min(sigma0_hh) =}, {np.max(sigma0_hh) =}")
    sigma0_hh[sigma0_hh<=0] = 0.
    sigma0_hh[sigma0_hh>1] = 1.

    sigma0_hv = s1[s1.get_band_number(band_id="sigma0_HV")]
    sigma0_hv = (np.log10(sigma0_hv) + 3.4) / 2.2
    sigma0_hv[sigma0_hv<0] = 0.
    sigma0_hv[sigma0_hv>1] = 1.

    noise_hv = s1[s1.get_band_number(band_id="noise_HV")]
    noise_hv_scaled = noise_hv / np.max(noise_hv)
    # noise_hv_flip = np.fliplr(noise_hv_scaled)
    # noise_hv_scaled[noise_hv_scaled < 0.15] = 0.

    for i in reversed(range(10)):
        noise_filename = self.img_dir+"input/"+img_name[:-4]+\
                         "_hh_noise_"+str(i)+".npy"
        if not os.path.isfile(noise_filename):
            noise_train = base_hh_img + ((i+1)/8.0) * noise_vec
            noise_train[noise_train<0] = 0.
            if i==9:
                max_val = np.max(noise_train)
            noise_train = noise_train / max_val
            np.save(noise_filename, noise_train)

            del noise_train


    # subtracted = sigma0_hv - noise_hv_scaled

    # increased = sigma0_hv + 0.5 * noise_hv_scaled
    # increased_flip = sigma0_hv + noise_hv_flip
    # # random_noise = sigma0_hv + (0.5 * np.random.normal(loc=0, scale=1, size=sigma0_hv.shape))
    # # random_noise[random_noise<0] = 0.
    # # random_noise[random_noise>1] = 1.
    #
    # noise_hv_scaled = noise_hv / np.max(noise_hv)
    # rolled_2k = np.empty_like(noise_hv_scaled)
    # rolled_2k[:,:2000] = noise_hv_scaled[:,-2000:]
    # rolled_2k[:,2000:] = noise_hv_scaled[:,:-2000]
    # roll_noise = sigma0_hv + rolled_2k
    # roll_noise[roll_noise<0] = 0.
    # roll_noise[roll_noise>1] = 1.
    #
    # rolled_3k = np.empty_like(noise_hv_scaled)
    # rolled_3k[:,:3500] = noise_hv_scaled[:,-3500:]
    # rolled_3k[:,3500:] = noise_hv_scaled[:,:-3500]
    # roll_noise_more = sigma0_hv + rolled_3k
    # roll_noise_more[roll_noise_more<0] = 0.
    # roll_noise_more[roll_noise_more>1] = 1.
    #
    # col = 1
    # while col < noise_hv_scaled.shape[1]:
    #     print("mean of col "+str(col)+" = "+str(np.mean(noise_hv_scaled[:,col])))
    #     col += 50

    # col = 0
    # while col < sigma0_hh.shape[1]:
    #     print("mean of col "+str(col)+" = "+str(np.mean(sigma0_hh[:,col])))
    #     col += 100

    # fig, ax = plt.subplots(1, 5, figsize=(15,10))
    # ax[0].imshow(sigma0_hv[::5,::5], vmax=1.00)
    # ax[1].imshow(increased[::5,::5], vmax=1.00)
    # ax[2].imshow(increased_flip[::5,::5], vmax=1.00)
    # ax[3].imshow(roll_noise[::5,::5], vmax=1.00)
    # ax[4].imshow(roll_noise_more[::5,::5], vmax=1.00)
    #
    # plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15,10))
    ax[0].imshow(sigma0_hv[::5,::5])
    ax[1].imshow(noise_hv_scaled[::5,::5])
    ax[2].imshow(subtracted[::5,::5])
    plt.show()
    plt.close()
