import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from s1denoise import Sentinel1Image
from skimage.filters import threshold_otsu
import pywt


img_list = ["S1A_EW_GRDM_1SDH_20181001T034748_20181001T034848_023937_029D19_B69F",
            "S1A_EW_GRDM_1SDH_20180901T030026_20180901T030125_023499_028EF3_42C5"]

for img in img_list:
    s1 = Sentinel1Image("../data/train/zip/"+img+".zip")
    sigma0_hv = s1[s1.get_band_number(band_id="sigma0_HV")]
    sigma0_hv = sigma0_hv[500:-500,500:-500]
    sigma0_hv = np.log10(sigma0_hv)
    print(f"{np.min(sigma0_hv) = }")
    print(f"{np.max(sigma0_hv) = }")
    sigma_scaled = (sigma0_hv - np.min(sigma0_hv)) / \
                   (np.max(sigma0_hv) - np.min(sigma0_hv))
    print(sigma_scaled[::5, ::5])
    _ = plt.hist(sigma_scaled[::5,::5], bins='auto')
    plt.show()
    plt.close()

    hv_denoised = s1.remove_texture_noise("HV")
    hv_denoised = hv_denoised[500:-500,500:-500]
    hv_denoised = np.log10(hv_denoised)
    print(f"{np.min(hv_denoised) = }")
    print(f"{np.max(hv_denoised) = }")
    hv_denoised_scaled = (hv_denoised - np.min(sigma0_hv)) / \
                         (np.max(sigma0_hv) - np.min(sigma0_hv))
    _ = plt.hist(hv_denoised[::5,::5], bins='auto')
    plt.show()
    plt.close()

    # fig = plt.figure(figsize=(20, 15))
    # ax = fig.add_subplot(2, 1, 1)
    # ax.imshow(sigma_scaled, interpolation="nearest")
    # ax = fig.add_subplot(2, 1, 2)
    # ax.imshow(hv_denoised_scaled, interpolation="nearest")
    # fig.tight_layout()
    # plt.show()
    # plt.close()

    # thresh = 0.2 * np.nanmax(hv_denoised_scaled)
    # coeff = pywt.wavedec(hv_denoised_scaled, "db4", mode="per")
    # coeff[1:] = (pywt.threshold(i, value=thresh, mode="hard") for i in coeff[1:])
    # recon = pywt.waverec(coeff, "db4", mode="per")

    # fig = plt.figure(figsize=(20, 15))
    # ax = fig.add_subplot(1, 7, 1)
    # ax.imshow(hv_denoised_scaled, interpolation="nearest")
    # ax = fig.add_subplot(1, 7, 2)
    # ax.imshow(cA_db, interpolation="nearest")
    # ax = fig.add_subplot(1, 7, 3)
    # ax.imshow(cA_db_level_5, interpolation="nearest")
    # ax = fig.add_subplot(1, 7, 4)
    # ax.imshow(cA_h, interpolation="nearest")
    # ax = fig.add_subplot(1, 7, 5)
    # ax.imshow(cA_haar_level_5, interpolation="nearest")
    # ax = fig.add_subplot(1, 7, 6)
    # ax.imshow(cA_mh, interpolation="nearest")
    # ax = fig.add_subplot(1, 7, 7)
    # ax.imshow(cA_mh_level_5, interpolation="nearest")
