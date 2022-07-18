import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from random import randrange
import random

from torch.utils.data import Dataset

class Sentinel1np(Dataset):

    def __init__(self, image_dir="../data/", dataset="train",
                 augmentation=False):
        self.dataset = dataset
        if self.dataset=="test":
            self.img_dir = image_dir+"test/"
        else:
            self.img_dir = image_dir+"train/"

        self.dim = 8192 # 4096

        if self.dataset != "train":
            self.augment = False
        else:
            self.augment = augmentation
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
        if not self.dataset=="eval":
            resize_list = [f for f in os.listdir(self.img_dir+"input/") \
                           if f[-8:]==str(self.dim)+".npy"]
            trf_list = [f for f in os.listdir(self.img_dir+"input/") \
                        if f[-8:]=="_trf.npy"]
            aug_list = [f for f in os.listdir(self.img_dir+"input/") \
                        if f[-12:-5]=="_noise_"]

            img_list_1 = [f for f in os.listdir(self.img_dir+"input/") \
                          if f not in resize_list]
            img_list_2 = [f for f in img_list_1 if f not in trf_list]
            img_list_3 = [f for f in img_list_2 if f not in aug_list]
            self.img_list = [f for f in img_list_3 if f[-4:]==".npy"]

            # resize saved imgs
            self._resize_images()

            # log transform imgs
            self._transform_images()

            # add noise to raw imgs
            self._augment_img()

        # return dataset
        input_img_list = []
        target_img_list = []

        self.img_list = [f for f in os.listdir(self.img_dir+"input/") \
                         if f[-8:]=="_trf.npy"]

        if self.dataset=="train":
            for img in self.img_list:
                target_img_filename = self.img_dir+"denoised/"+img
                target_img = np.load(target_img_filename)
                if np.isnan(target_img).any():
                    print("image has NaN values present - "+\
                          "not added to training set")
                    continue
                else:
                    del target_img
                if self.augment:
                    range_max = 3
                else:
                    range_max = 1
                for i in range(range_max):
                    train_img_filename = self.img_dir+"input/"+img[:-4]+\
                                         "_noise_"+str(i)+".npy"
                    input_img_list.append(train_img_filename)
                    target_img_list.append(target_img_filename)

        elif self.dataset=="eval":
            for img in self.img_list[:10]:
                target_img_filename = self.img_dir+"denoised/"+img
                target_img = np.load(target_img_filename)
                if np.isnan(target_img).any():
                    print("image has NaN values present - "+\
                          "not added to training set")
                    continue
                else:
                    del target_img
                train_img_filename = self.img_dir+"input/"+img[:-4]+"_noise"+\
                                     "_0.npy"
                input_img_list.append(train_img_filename)
                target_img_list.append(target_img_filename)

        else:
            for img in self.img_list:
                target_img_filename = self.img_dir+"denoised/"+img
                target_img = np.load(target_img_filename)
                if np.isnan(target_img).any():
                    print("image has NaN values present - "+\
                          "not added to training set")
                    continue
                else:
                    del target_img
                train_img_filename = self.img_dir+"input/"+img[:-4]+"_noise"+\
                                     "_0.npy"
                input_img_list.append(train_img_filename)
                target_img_list.append(target_img_filename)

        assert len(input_img_list) == len(target_img_list)
        return input_img_list, target_img_list


    def _resize_images(self):
        print("---Resizing images to "+str(self.dim)+" pixels---")

        resized_imgs = [f for f in os.listdir(self.img_dir+"input/") \
                        if f[-12:]=="_sz_"+str(self.dim)+".npy"]

        if len(resized_imgs)==len(self.img_list):
            print("Already resized")
            return

        for img in self.img_list:
            raw_filename = self.img_dir+"input/"+img
            noise_filename = self.img_dir+"noise_vec/"+img
            denoised_filename = self.img_dir+"denoised/"+img

            resized_raw_filename = raw_filename[:-4]+"_sz_"+str(self.dim)+\
                                   ".npy"
            resized_noise_filename = noise_filename[:-4]+"_sz_"+\
                                     str(self.dim)+".npy"
            resized_denoised_filename = denoised_filename[:-4]+"_sz_"+\
                                        str(self.dim)+".npy"

            if os.path.isfile(resized_raw_filename):
                print("Skipping")
                continue
            else:
                raw = np.load(raw_filename)
                noise = np.load(noise_filename)
                denoised = np.load(denoised_filename)

                start_ind_row = int((raw.shape[0] - self.dim) / 2)
                start_ind_col = int((raw.shape[0] - self.dim) / 2)

                raw = raw[start_ind_row:start_ind_row+self.dim,
                          start_ind_col:start_ind_col+self.dim]
                noise = noise[start_ind_row:start_ind_row+self.dim,
                              start_ind_col:start_ind_col+self.dim]
                denoised = denoised[start_ind_row:start_ind_row+self.dim,
                                    start_ind_col:start_ind_col+self.dim]

                if raw.shape[0]<self.dim:
                    print("File not square - deleting image")
                    os.remove(self.img_dir+"input/"+img)
                    continue

                np.save(resized_raw_filename, raw)
                np.save(resized_noise_filename, noise)
                np.save(resized_denoised_filename, denoised)


    def _transform_images(self):
        print("---Transforming images---")
        self.img_list = [f for f in os.listdir(self.img_dir+"input/") \
                         if f[-12:]=="_sz_"+str(self.dim)+".npy"]

        trf_list = [f for f in os.listdir(self.img_dir+"input/") \
                    if f[-8:]=="_trf.npy"]
        if len(trf_list)==len(self.img_list):
            print("Already transformed")
            return

        for img in self.img_list:
            resized_raw_filename = self.img_dir+"input/"+img
            resized_noise_filename = self.img_dir+"noise_vec/"+img
            resized_denoised_filename = self.img_dir+"denoised/"+img

            transform_raw_filename = resized_raw_filename[:-4]+"_trf.npy"
            transform_noise_filename = resized_noise_filename[:-4]+"_trf.npy"
            transform_denoised_filename = resized_denoised_filename[:-4]+\
                                          "_trf.npy"

            if not os.path.isfile(transform_raw_filename):
                img = np.load(resized_raw_filename)
                img = (np.log10(img) + 3.4) / 2.2
                img[img<0] = 0
                img[img>1.] = 1.
                np.save(transform_raw_filename, img)

            if not os.path.isfile(transform_noise_filename):
                noise_img = np.load(resized_noise_filename)
                noise_img_scaled = noise_img / np.max(noise_img)
                np.save(transform_noise_filename, noise_img_scaled)

            if not os.path.isfile(transform_denoised_filename):
                denoise_img = np.load(resized_denoised_filename)
                denoise_img = (np.log10(denoise_img) + 3.4) / 2.2
                denoise_img[denoise_img<0] = 0
                denoise_img[denoise_img>1.] = 1.
                np.save(transform_denoised_filename, denoise_img)


    def _augment_img(self):
        print("---Doing Data Augmentation---")
        self.img_list = [f for f in os.listdir(self.img_dir+"input/") \
                         if f[-8:]=="_trf.npy"]

        aug_list = [f for f in os.listdir(self.img_dir+"input/") \
                    if f[-6:]=="_0.npy"]
        if len(aug_list)==len(self.img_list):
            print("Augmented Data already saved")
            return

        for img_name in self.img_list:
            raw_train_filename = self.img_dir+"input/"+img_name
            noise_vec_filename = self.img_dir+"noise_vec/"+img_name

            base_img = np.load(raw_train_filename)
            noise_vec = np.load(noise_vec_filename)

            for i in reversed(range(1)):
            # for i in reversed(range(3)):
                noisy_filename = self.img_dir+"input/"+img_name[:-4]+\
                                 "_noise_"+str(i)+".npy"
                if os.path.isfile(noisy_filename):
                    continue
                else:
                    if i > 0:
                        noise_train = (base_img) + ((i)/10.) * noise_vec
                    else:
                        noise_train = base_img
                    noise_train[noise_train<0] = 0.
                    noise_train[noise_train>1] = 1.
                    np.save(noisy_filename, noise_train)
                    del noise_train


    def save_training_img(self):
        if self.dataset=="train":
            print("---Plotting Training Images---")
            image_dir = self.img_dir+"input/images/"

            if not os.path.isdir(image_dir):
                os.mkdir(image_dir)

            plot_list = [f for f in os.listdir(image_dir) if f[-4:]==".pdf"]
            if len(plot_list)==len(self.img_list):
                print("Training Images already saved")
                return

            for i in range(len(self.img_list)):
                img = self.img_list[i]
                print("Plotting img: "+img)
                img_filename = self.img_dir+"input/"+img[:-4]

                fig, ax = plt.subplots(1, 3, figsize=(20, 15))

                data_0 = np.load(img_filename+"_noise_0.npy")
                ax[0].imshow(data_0)
                ax[0].set_title("no noise")

                data_1 = np.load(img_filename+"_noise_1.npy")
                ax[1].imshow(data_1)
                ax[1].set_title("noise 1")

                target = np.load(self.img_dir+"denoised/"+img)
                ax[2].imshow(target)
                ax[2].set_title("train target")
                del data_0, data_1, target

                if not os.path.isfile(image_dir+img[:-4]+".pdf"):
                    plt.savefig(image_dir+img[:-4]+".pdf")
                # plt.show()
                plt.close()




if __name__ == "__main__":
    # change this to a sys arg
    DATA_PATH = "/home/benjamin/Dropbox/nerd_stuff/postdoc/code/denoising/"+\
                "images/test/"
    my_data = Sentinel1Test(DATA_PATH)
