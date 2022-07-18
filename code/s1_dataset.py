import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from random import randrange

from s1denoise.sentinel1image import Sentinel1Image
from torch.utils.data import Dataset


class Sentinel1(Dataset):

    def __init__(self, image_dir="../data/", dataset="train"):
        self.dataset = dataset
        if self.dataset=="test":
            self.img_dir = image_dir+"test/"
        else:
            self.img_dir = image_dir+"train/"
        self.img_list = [f for f in os.listdir(self.img_dir+"zip/") \
                         if f[-3:]=="zip"]
        self.dim = 4096 # 8192
        # self.raw, self.denoised = self._get_data()
        self.train_list, self.target_list = self._get_data()
        if self.dataset=="train":
            print("No training imgs:", self.__len__())
            print("Image dimension:", self.dim)
        else:
            print("No imgs to pred:", self.__len__())


    def __len__(self):
        return len(self.train_list)


    def __getitem__(self, index):
        train_img = np.load(self.train_list[index]).astype(np.float32)
        train_img = train_img.reshape(train_img.shape[0],
                                      train_img.shape[1],
                                      1)
        train_img = np.swapaxes(train_img, 2, 0)
        train_img = torch.from_numpy(train_img)

        target_img = np.load(self.target_list[index]).astype(np.float32)
        target_img = target_img.reshape(target_img.shape[0],
                                        target_img.shape[1],
                                        1)
        target_img = np.swapaxes(target_img, 2, 0)
        target_img = torch.from_numpy(target_img)

        return train_img, target_img


    def _get_data(self):

        aug_img = self.img_dir+"input/"+self.img_list[0][:-4]+"_raw.npy"
        if not os.path.isfile(aug_img):

            trf_img = self.img_dir+"noise_vec/"+self.img_list[0][:-4]+\
                      "_trf.npy"
            if not os.path.isfile(trf_img):

                resize_img = self.img_dir+"input/"+self.img_list[0][:-4]+\
                             "_resized.npy"
                if not os.path.isfile(resize_img):

                    # if raw images exist
                    raw_img = self.img_dir+"input/"+self.img_list[0][:-4]+\
                              ".npy"
                    if not os.path.isfile(raw_img):

                        # extract raw from zip AND save
                        self._extract_from_zip()

                    self.img_list = [f for f in os.listdir(self.img_dir+"zip/")\
                                     if f[-3:]=="zip"]
                    # resize saved imgs
                    self._resize_images()

                self.img_list = [f for f in os.listdir(self.img_dir+"zip/") \
                                 if f[-3:]=="zip"]
                # log transform imgs
                self._transform_images()

            self.img_list = [f for f in os.listdir(self.img_dir+"zip/") \
                             if f[-3:]=="zip"]
            # add noise to raw imgs
            self._augment_img()

        # return dataset
        input_img_list = []
        target_img_list = []

        self.img_list = [f for f in os.listdir(self.img_dir+"zip/") \
                         if f[-3:]=="zip"]

        if self.dataset=="train":
            for img in self.img_list:
                target_img_filename = self.img_dir+"denoised/"+img[:-4]+".npy"
                target_img = np.load(target_img_filename)
                if np.isnan(target_img).any():
                    print("image has NaN values present - "+\
                          "not added to training set")
                    continue
                else:
                    del target_img
                img_types = ["noise_0", "noise_1", "noise_2", "noise_3",
                             "noise_4", "noise_5", "noise_6", "noise_7"]
                # img_types = ["noise_0"]
                for type in img_types:
                    train_img_filename = self.img_dir+"input/"+img[:-4]+"_"+\
                                         type+".npy"
                    input_img_list.append(train_img_filename)
                    target_img_list.append(target_img_filename)

        else:
            for img in self.img_list:
                target_img_filename = self.img_dir+"denoised/"+img[:-4]+".npy"
                target_img = np.load(target_img_filename)
                if np.isnan(target_img).any():
                    print("image has NaN values present - "+\
                          "not added to training set")
                    continue
                else:
                    del target_img
                noise_img_filename = self.img_dir+"input/"+img[:-4]+"_raw.npy"
                input_img_list.append(noise_img_filename)
                target_img_list.append(target_img_filename)

        assert len(input_img_list) == len(target_img_list)
        return input_img_list, target_img_list


    def _extract_from_zip(self):
        input_dir = self.img_dir+"input/"
        if not os.path.isdir(input_dir):
            os.mkdir(input_dir)

        noise_dir = self.img_dir+"noise_vec/"
        if not os.path.isdir(noise_dir):
            os.mkdir(noise_dir)

        denoised_dir = self.img_dir+"denoised/"
        if not os.path.isdir(denoised_dir):
            os.mkdir(denoised_dir)

        for no, img in enumerate(self.img_list):
            print("Image #"+str(no)+" of "+str(len(self.img_list)))
            raw_filename = input_dir+img[:-4]+".npy"
            noise_filename = noise_dir+img[:-4]+".npy"
            denoised_filename = denoised_dir+img[:-4]+".npy"

            if not os.path.isfile(raw_filename):
                image_path = self.img_dir+"zip/"+img
                try:
                    s1 = Sentinel1Image(image_path)
                except:
                    print("image doesn't work")
                    os.remove(image_path)
                    continue
                sigma0_hv = s1[s1.get_band_number(band_id="sigma0_HV")]
                np.save(raw_filename, sigma0_hv)

                noise_hv = s1[s1.get_band_number(band_id="noise_HV")]
                np.save(noise_filename, noise_hv)

                sigma0_hv_denoised = s1.remove_texture_noise("HV")
                np.save(denoised_filename, sigma0_hv_denoised)


    def _resize_images(self):
        print("---Resizing images to "+str(self.dim)+" pixels---")

        for img in self.img_list:
            raw_filename = self.img_dir+"input/"+img[:-4]+".npy"
            noise_filename = self.img_dir+"noise_vec/"+img[:-4]+".npy"
            denoised_filename = self.img_dir+"denoised/"+img[:-4]+".npy"

            resized_raw_filename = raw_filename[:-4]+"_resized.npy"
            resized_noise_filename = noise_filename[:-4]+"_resized.npy"
            resized_denoised_filename = denoised_filename[:-4]+"_resized.npy"

            if not os.path.isfile(resized_raw_filename):
                raw = np.load(raw_filename)
                noise = np.load(noise_filename)
                denoised = np.load(denoised_filename)

                start_ind_row = int((raw.shape[0] - self.dim) / 2)
                start_ind_col = 200

                raw = raw[start_ind_row:start_ind_row+self.dim,
                          start_ind_col:start_ind_col+self.dim]
                noise = noise[start_ind_row:start_ind_row+self.dim,
                              start_ind_col:start_ind_col+self.dim]
                denoised = denoised[start_ind_row:start_ind_row+self.dim,
                                    start_ind_col:start_ind_col+self.dim]

                if raw.shape[0]<self.dim:
                    print("File not square - deleting image")
                    os.remove(self.img_dir+"zip/"+img)
                    continue

                np.save(resized_raw_filename, raw)
                np.save(resized_noise_filename, noise)
                np.save(resized_denoised_filename, denoised)

                if os.path.isfile(raw_filename):
                    os.remove(raw_filename)
                if os.path.isfile(noise_filename):
                    os.remove(noise_filename)
                if os.path.isfile(denoised_filename):
                    os.remove(denoised_filename)


    def _transform_images(self):
        print("---Transforming images---")

        for img in self.img_list:
            resized_raw_filename = self.img_dir+"input/"+img[:-4]+\
                                   "_resized.npy"
            resized_noise_filename = self.img_dir+"noise_vec/"+img[:-4]+\
                                   "_resized.npy"
            resized_denoised_filename = self.img_dir+"denoised/"+img[:-4]+\
                                        "_resized.npy"

            transform_raw_filename = resized_raw_filename[:-12]+"_raw.npy"
            transform_noise_filename = resized_noise_filename[:-12]+"_trf.npy"
            transform_denoised_filename = resized_denoised_filename[:-12]+\
                                          ".npy"

            if not os.path.isfile(transform_raw_filename):
                img = np.load(resized_raw_filename)
                img = (np.log10(img) + 3.4) / 2.2
                img[img<0] = 0
                img[img>1] = 1
                np.save(transform_raw_filename, img)
            if os.path.isfile(resized_raw_filename):
                os.remove(resized_raw_filename)

            if not os.path.isfile(transform_noise_filename):
                noise_img = np.load(resized_noise_filename)
                noise_img_scaled = noise_img / np.max(noise_img)
                np.save(transform_noise_filename, noise_img_scaled)
            if os.path.isfile(resized_noise_filename):
                os.remove(resized_noise_filename)

            if not os.path.isfile(transform_denoised_filename):
                denoise_img = np.load(resized_denoised_filename)
                denoise_img = (np.log10(denoise_img) + 3.4) / 2.2
                denoise_img[denoise_img<0] = 0
                denoise_img[denoise_img>1.] = 1.
                np.save(transform_denoised_filename, denoise_img)
            if os.path.isfile(resized_denoised_filename):
                os.remove(resized_denoised_filename)


    def _augment_img(self):
        if self.dataset=="train":
            for img_name in self.img_list:
                raw_train_filename = self.img_dir+"input/"+\
                                     img_name[:-4]+"_raw.npy"
                noise_vec_filename = self.img_dir+"noise_vec/"+img_name[:-4]+\
                                     "_trf.npy"

                base_img = np.load(raw_train_filename)
                noise_vec = np.load(noise_vec_filename)

                for i in reversed(range(8)):
                    noisy_filename = self.img_dir+"input/"+img_name[:-4]+\
                                     "_noise_"+str(i)+".npy"
                    if not os.path.isfile(noisy_filename):
                        noise_train = (base_img) + ((i)/8.) * noise_vec
                        noise_train[noise_train<0] = 0.
                        noise_train[noise_train>1] = 1.
                        np.save(noisy_filename, noise_train)
                        del noise_train

                if os.path.isfile(noise_vec_filename):
                    os.remove(noise_vec_filename)

        else:
            for img_name in self.img_list:
                noise_vec_filename = self.img_dir+"noise_vec/"+img_name[:-4]+\
                                     "_trf.npy"
                if os.path.isfile(noise_vec_filename):
                    os.remove(noise_vec_filename)

        if os.path.isdir(self.img_dir+"noise_vec/"):
            os.rmdir(self.img_dir+"noise_vec/")


    def save_training_img(self):

        if self.dataset=="train":
            image_dir = self.img_dir+"input/images/"
            if not os.path.isdir(image_dir):
                os.mkdir(image_dir)

            for i in range(len(self.img_list)):
                img = self.img_list[i]
                img_filename = self.img_dir+"input/"+img[:-4]

                fig, ax = plt.subplots(2, 3, figsize=(20, 15))

                data_1 = np.load(img_filename+"_noise_1.npy")
                ax[0,0].imshow(data_1)
                ax[0,0].set_title("noise 1")
                data_3 = np.load(img_filename+"_noise_3.npy")
                ax[0,1].imshow(data_3)
                ax[0,1].set_title("noise 3")
                data_5 = np.load(img_filename+"_noise_5.npy")
                ax[0,2].imshow(data_5)
                ax[0,2].set_title("noise 5")
                data_7 = np.load(img_filename+"_noise_7.npy")
                ax[1,0].imshow(data_7)
                ax[1,0].set_title("noise 7")

                raw = np.load(img_filename+"_raw.npy")
                ax[1,1].imshow(raw)
                ax[1,1].set_title("raw")
                target = np.load(self.img_dir+"denoised/"+img[:-4]+".npy")
                ax[1,2].imshow(target)
                ax[1,2].set_title("train target")
                del data_1, data_3, data_5, data_7, raw, target

                if not os.path.isfile(image_dir+img[:-4]+".pdf"):
                    plt.savefig(image_dir+img[:-4]+".pdf")
                # plt.show()
                plt.close()



    def save_predicted_img(self):
        if self.dataset=="test":
            pred_image_dir = "../predictions/test_skip/images/"
            if not os.path.isdir(pred_image_dir):
                os.mkdir(pred_image_dir)

            for i in range(len(self.img_list)):
                img = self.img_list[i]

                raw = np.load(self.img_dir+"input/"+img[:-4]+"_raw.npy")
                denoised = np.load(self.img_dir+"denoised/"+img[:-4]+".npy")
                pred_raw = np.load("../predictions/test_skip/"+img[:-4]+\
                                   "_pred_raw.npy")
                pred_denoise = np.load("../predictions/test_skip/"+img[:-4]+\
                                       "_pred_denoise.npy")

                fig, ax = plt.subplots(1, 4, figsize=(20,12))
                ax[0].imshow(raw)
                ax[0].set_title("raw - noisy")
                ax[1].imshow(pred_raw)
                ax[1].set_title("denoised from raw img")
                ax[2].imshow(denoised)
                ax[2].set_title("nansen denoised")
                ax[3].imshow(pred_denoise)
                ax[3].set_title("denoised from nansen img")
                # plt.show()
                if not os.path.isfile(pred_image_dir+img[:-4]+".pdf"):
                    plt.savefig(pred_image_dir+img[:-4]+".pdf")
                plt.close()


if __name__ == "__main__":
    # change this to a sys arg
    DATA_PATH = "/home/benjamin/Dropbox/nerd_stuff/postdoc/code/denoising/"+\
                "images/test/"
    my_data = Sentinel1Test(DATA_PATH)
