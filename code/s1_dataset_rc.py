import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from random import randrange
import random

from torch.utils.data import Dataset

class Sentinel1(Dataset):

    def __init__(self, image_dir="../data/", dataset="train", dimension=4096):
        self.dataset = dataset
        print(f"{self.dataset =}")
        if self.dataset=="test":
            self.img_dir = image_dir+"test/"
        else:
            self.img_dir = image_dir+"train/"

        self.dim = dimension

        self.train_list, self.target_list = self._get_data()
        if self.dataset=="train":
            print("No training imgs:", self.__len__())
            print("Image dimension:", self.dim)
        else:
            print("No imgs to pred:", self.__len__())
            print("Image dimension:", self.dim)


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
        input_img_list = []
        target_img_list = []
        self.img_list = [f for f in os.listdir(self.img_dir+"input/") \
                         if f[-11:]=="noise_0.npy"]
        if self.dataset!="eval":
            for img in self.img_list:
                target_img_filename = self.img_dir+"denoised/"+img[:-12]+".npy"
                target_img = np.load(target_img_filename)
                if np.isnan(target_img).any():
                    print(f"{img = }")
                    print("Target image has NaN values present - "+\
                          "not added to training set")
                    continue
                else:
                    del target_img
                train_img_filename = self.img_dir+"input/"+img
                train_img = np.load(train_img_filename)
                if np.isnan(train_img).any():
                    print(f"{img = }")
                    print("Train image has NaN values present - "+\
                          "not added to training set")
                    continue
                else:
                    del train_img
                input_img_list.append(train_img_filename)
                target_img_list.append(target_img_filename)

        else:
            if len(self.img_list) > 10:
                self.img_list = self.img_list[:10]
            for img in self.img_list:
                target_img_filename = self.img_dir+"denoised/"+img[:-12]+ \
                                      ".npy"
                target_img = np.load(target_img_filename)
                if np.isnan(target_img).any():
                    print(f"{img = }")
                    print("Target image has NaN values present - "+\
                          "not added to training set")
                    continue
                else:
                    del target_img
                train_img_filename = self.img_dir+"input/"+img
                train_img = np.load(train_img_filename)
                if np.isnan(train_img).any():
                    print(f"{img = }")
                    print("Train image has NaN values present - "+\
                          "not added to training set")
                    continue
                else:
                    del train_img
                input_img_list.append(train_img_filename)
                target_img_list.append(target_img_filename)

        assert len(input_img_list) == len(target_img_list)
        return input_img_list, target_img_list


    def save_training_img(self):
        if self.dataset=="train":
            print("---Plotting Training Images---")
            plot_dir = self.img_dir+"plots/"
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)
            train_plot_dir = plot_dir + "train/"
            if not os.path.isdir(train_plot_dir):
                os.mkdir(train_plot_dir)

            plot_list = [f for f in os.listdir(train_plot_dir) \
                         if f[-4:]==".pdf"]
            print("len plot list:", len(plot_list))
            print("data self len:", self.__len__())
            if len(plot_list)==self.__len__():
                print("Training Images already saved")
                return

            for img in self.img_list:
                print("Plotting img: "+img)
                fig, ax = plt.subplots(1, 2, figsize=(20, 15))

                data = np.load(self.img_dir+"input/"+img)
                ax[0].imshow(data)
                ax[0].set_title("noisy / raw")

                target = np.load(self.img_dir+"denoised/"+img[:-12]+".npy")
                ax[1].imshow(target)
                ax[1].set_title("target (nansen denoised)")
                del data, target

                if not os.path.isfile(train_plot_dir+img[:-4]+".pdf"):
                    plt.savefig(train_plot_dir+img[:-4]+".pdf")
                plt.close()


if __name__ == "__main__":
    # change this to a sys arg
    DATA_PATH = "/home/benjamin/Dropbox/nerd_stuff/postdoc/code/denoising/"+ \
                "images/test/"
    my_data = Sentinel1Test(DATA_PATH)
