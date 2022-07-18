import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn.functional as F


class StridedConvAutoencoder(nn.Module):
    def __init__(self):
        super(StridedConvAutoencoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.LeakyReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.LeakyReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.LeakyReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.LeakyReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.LeakyReLU()
        )


        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
            nn.LeakyReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
            nn.LeakyReLU()
        )

        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.LeakyReLU(),
        )

        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.LeakyReLU(),
        )

        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.LeakyReLU(),
        )

        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.LeakyReLU(),
        )

        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.LeakyReLU(),
        )

        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=32,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.LeakyReLU(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=33,
                      out_channels=1,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
            nn.Sigmoid(),
        )


    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        conv7_out = self.conv7(conv6_out)
        conv8_out = self.conv8(conv7_out)
        tconv6_out = self.tconv6(conv8_out)
        tconv5_out = self.tconv5(tconv6_out)
        tconv4_out = self.tconv4(tconv5_out)
        tconv4_and_skip = torch.cat([tconv4_out, conv3_out], 1)
        tconv3_out = self.tconv3(tconv4_and_skip)
        tconv3_and_skip = torch.cat([tconv3_out, conv2_out], 1)
        tconv2_out = self.tconv2(tconv3_and_skip)
        tconv2_and_skip = torch.cat([tconv2_out, conv1_out], 1)
        tconv1_out = self.tconv1(tconv2_and_skip)
        tconv1_and_skip = torch.cat([tconv1_out, x], 1)
        out = self.final(tconv1_and_skip)

        # print(f"{x.shape = }")
        # print(f"{conv1_out.shape = }")
        # print(f"{conv2_out.shape = }")
        # print(f"{conv3_out.shape = }")
        # print(f"{conv4_out.shape = }")
        # print(f"{conv5_out.shape = }")
        # print(f"{conv6_out.shape = }")
        # print(f"{conv7_out.shape = }")
        # print(f"{conv8_out.shape = }")
        # print(f"{tconv6_out.shape = }")
        # print(f"{tconv5_out.shape = }")
        # print(f"{tconv4_out.shape = }")
        # print(f"{tconv3_out.shape = }")
        # print(f"{tconv2_out.shape = }")
        # print(f"{tconv1_out.shape = }")
        # print(f"{out.shape = }")

        return out



    @staticmethod
    def train_model(model, train_data, loss_fn=nn.MSELoss(), n_epochs=20):
        start = time.time()
        model.float()
        model_dir = "../models/"
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_filepath = model_dir+"ae_8192.pth"
        if os.path.isfile(model_filepath):
            model = torch.load(model_filepath)
            print("trained model already found")
            return model

        already_completed_epochs = n_epochs
        temp_model_filepath = model_dir+"ae_8192_epoch_"+\
                              str(already_completed_epochs)+".pth"
        while not os.path.isfile(temp_model_filepath) and \
                  already_completed_epochs > 0:
            already_completed_epochs -= 1
            temp_model_filepath = model_dir+"ae_8192_epoch_"+\
                                  str(already_completed_epochs)+".pth"

        if already_completed_epochs>0:
            del model
            model = torch.load(temp_model_filepath)

        # set loss and optimizer
        criterion = loss_fn
        optimizer = optim.Adam(model.parameters())

        # create torch data loader
        train_loader = data.DataLoader(train_data, batch_size=1, shuffle=True)

        min_train_loss = 1000000.

        # train model
        print("training model...")
        for epoch in range(already_completed_epochs+1, n_epochs+1):
            epoch_start = time.time()
            train_loss = 0.0
            for instance in train_loader:
                noisy, clean = instance
                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                # forward pass
                outputs = model(noisy)

                # calculate the loss
                loss = criterion(outputs, clean)

                # backprop
                loss.backward()

                # update parameters
                optimizer.step()

                # update training loss
                train_loss += loss.item()*noisy.size(0)

            # print avg training loss
            train_loss = train_loss/len(train_loader)
            print("Epoch: "+str(epoch)+", Training Loss:"+str(train_loss),
                  "Time: "+str((time.time()-epoch_start)/60)+"min")

            if epoch < n_epochs:
                torch.save(model, model_dir+"ae_8192_epoch_"+str(epoch)+".pth")
            # prior_filepath = model_dir+"ae_8192_epoch_"+str(epoch-1)+".pth"
            # if epoch > 1 and os.path.isfile(prior_filepath):
            #     os.remove(prior_filepath)

        torch.save(model, model_dir+"ae_8192.pth")

        return model


    @staticmethod
    def validate_on_train_img(model, eval_data):
        print("--- validation predictions ---")
        eval_loader = data.DataLoader(eval_data, batch_size=1, shuffle=False)

        predictions_dir = "../predictions/eval/"
        if not os.path.isdir(predictions_dir):
            os.mkdir(predictions_dir)
        pred_image_dir = predictions_dir+"/images/"
        if not os.path.isdir(pred_image_dir):
            os.mkdir(pred_image_dir)

        model.eval()
        with torch.no_grad():
            for i, img in enumerate(eval_loader):
                raw, _ = img
                prediction_raw_filename = predictions_dir+\
                                          eval_data.img_list[i][:-4]+\
                                          "_pred_raw.npy"
                if os.path.isfile(prediction_raw_filename):
                    continue
                pred = model(raw)
                del raw
                pred = np.transpose(pred.detach().numpy()[0][0])
                np.save(prediction_raw_filename, pred)

                input = np.load(eval_data.img_dir+"input/"+\
                                eval_data.img_list[i])
                output = np.load(eval_data.img_dir+"denoised/"+\
                                 eval_data.img_list[i][:-12]+".npy")

                fig, ax = plt.subplots(1, 3, figsize=(20,12))
                ax[0].imshow(input)
                ax[0].set_title("raw")
                ax[1].imshow(output)
                ax[1].set_title("them denoised")
                ax[2].imshow(pred)
                ax[2].set_title("us denoised")

                if not os.path.isfile(pred_image_dir+eval_data.img_list[i][:-4]+\
                                      ".pdf"):
                    plt.savefig(pred_image_dir+eval_data.img_list[i][:-4]+".pdf")
                plt.close()

    @staticmethod
    def make_predictions(model, test_data):

        test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False)

        predictions_dir = "../predictions/test/"
        if not os.path.isdir(predictions_dir):
            os.mkdir(predictions_dir)

        model.eval()
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                raw_pred_filename = predictions_dir+test_data.img_list[i][:-4]+\
                                      "_pred_raw.npy"
                denoise_pred_filename = predictions_dir+test_data.img_list[i][:-4]+\
                                      "_pred_denoise.npy"
                if os.path.isfile(raw_pred_filename) and \
                   os.path.isfile(denoise_pred_filename):
                    continue

                raw, denoised = img

                raw_prediction = model(raw)
                raw_prediction = raw_prediction.detach().numpy()[0][0]
                raw_prediction = np.transpose(raw_prediction)
                np.save(raw_pred_filename, raw_prediction)

                denoise_prediction = model(denoised)
                denoise_prediction = denoise_prediction.detach().numpy()[0][0]
                denoise_prediction = np.transpose(denoise_prediction)
                np.save(denoise_pred_filename, denoise_prediction)

        print("Predictions saved")

        pred_image_dir = "../predictions/test/images/"
        if not os.path.isdir(pred_image_dir):
            os.mkdir(pred_image_dir)

        for img in test_data.img_list:
            fig, ax = plt.subplots(2, 4, figsize=(20,12))

            raw = np.load(test_data.img_dir+"input/"+img)
            ax[0,0].imshow(raw)
            ax[0,0].set_title("raw image")
            raw_mean = np.mean(raw, axis=0)
            ax[1,0].plot(raw_mean)
            del raw, raw_mean

            pred_raw = np.load("../predictions/test/"+img[:-4]+\
                               "_pred_raw.npy")
            ax[0,1].imshow(pred_raw)
            ax[0,1].set_title("model output (from raw)")
            pred_raw_mean = np.mean(pred_raw, axis=0)
            ax[1,1].plot(pred_raw_mean)
            del pred_raw, pred_raw_mean

            nansen = np.load(test_data.img_dir+"denoised/"+img[:-12]+".npy")
            ax[0,2].imshow(nansen)
            ax[0,2].set_title("nansen denoised")
            nansen_mean = np.mean(nansen, axis=0)
            ax[1,2].plot(nansen_mean)
            del nansen, nansen_mean

            pred_denoise = np.load("../predictions/test/"+img[:-4]+\
                                   "_pred_denoise.npy")
            ax[0,3].imshow(pred_denoise)
            ax[0,3].set_title("model output (from nansen)")
            pred_nansen_mean = np.mean(pred_denoise, axis=0)
            ax[1,3].plot(pred_nansen_mean)
            del pred_denoise, pred_nansen_mean

            ax10 = ax[1,0]
            ax[1,1].sharey(ax10)
            ax[1,2].sharey(ax10)
            ax[1,3].sharey(ax10)
            if not os.path.isfile(pred_image_dir+img[:-4]+".pdf"):
                plt.savefig(pred_image_dir+img[:-4]+".pdf")
            plt.close()


    @staticmethod
    def postprocessing(test_dir):

        kernel_size_list = [32, 1024, 2048]

        # create test_img_list from test_dir
        test_img_list = [f for f in os.listdir(test_dir+"input/") \
                         if f[-8:]=="_trf.npy"]

        # predictions_dir = "../predictions/test/"
        predictions_dir = "../predictions/test_all_images/"

        for img in test_img_list[:2]:
            smoothed_out = []
            raw_pred_filename = predictions_dir+img[:-4]+"_pred_raw.npy"
            raw_pred = np.load(raw_pred_filename)
            for size in kernel_size_list:
                smoothed_out.append(StridedConvAutoencoder._averaging(raw_pred,
                                                                      size))
            smoothed_out_np = np.array(smoothed_out)
            smoothed_out_np = np.mean(smoothed_out_np, axis=0)

            pred_smooth_dir = "../predictions/test_smooth/"
            if not os.path.isdir(pred_smooth_dir):
                os.mkdir(pred_smooth_dir)
            # if not os.path.isfile(pred_smooth_dir+img[:-4]+"_pred_raw.npy"):
            #     np.save(pred_smooth_dir+img[:-4]+"_pred_raw.npy", smoothed_out_np)

            denoised_smoothed_out = []
            denoise_pred_filename = predictions_dir+img[:-4]+"_pred_denoise.npy"
            denoise_pred = np.load(denoise_pred_filename)
            for size in kernel_size_list:
                denoised_smoothed_out.append(StridedConvAutoencoder._averaging(denoise_pred,
                                                                               size))
            denoised_smoothed_out_np = np.array(denoised_smoothed_out)
            denoised_smoothed_out_np = np.mean(denoised_smoothed_out_np, axis=0)


            # if not os.path.isfile(pred_smooth_dir+img[:-4]+"_pred_denoise.npy"):
            #     np.save(pred_smooth_dir+img[:-4]+"_pred_denoise.npy", denoise_out)

            # plot
            fig, ax = plt.subplots(4, 3, figsize=(20,20))
            # fig, ax = plt.subplots(4, 3)

            raw = np.load(test_dir+"/input/"+img[:-4]+"_noise_0.npy")
            ax[0,0].imshow(raw)
            ax[0,0].set_title("raw image")
            raw_mean = np.mean(raw, axis=0)
            ax[1,0].plot(raw_mean)
            del raw, raw_mean

            ax[0,1].imshow(raw_pred)
            ax[0,1].set_title("model output (from raw)")
            pred_raw_mean = np.mean(raw_pred, axis=0)
            ax[1,1].plot(pred_raw_mean)
            del raw_pred, pred_raw_mean

            ax[0,2].imshow(smoothed_out_np)
            ax[0,2].set_title("model output smoothed")
            smooth_raw_mean = np.mean(smoothed_out_np, axis=0)
            ax[1,2].plot(smooth_raw_mean)
            del smoothed_out_np, smooth_raw_mean

            nansen = np.load(test_dir+"/denoised/"+img)
            ax[2,0].imshow(nansen)
            ax[2,0].set_title("nansen denoised")
            nansen_mean = np.mean(nansen, axis=0)
            ax[3,0].plot(nansen_mean)
            del nansen, nansen_mean

            ax[2,1].imshow(denoise_pred)
            ax[2,1].set_title("model output (from nansen)")
            pred_nansen_mean = np.mean(denoise_pred, axis=0)
            ax[3,1].plot(pred_nansen_mean)
            del denoise_pred, pred_nansen_mean

            ax[2,2].imshow(denoised_smoothed_out_np)
            ax[2,2].set_title("model output smoothed")
            smooth_denoise_mean = np.mean(denoised_smoothed_out_np, axis=0)
            ax[3,2].plot(smooth_denoise_mean)
            del denoised_smoothed_out_np, smooth_denoise_mean

            ax10 = ax[1,0]
            ax[1,1].sharey(ax10)
            ax[1,2].sharey(ax10)
            ax[3,0].sharey(ax10)
            ax[3,1].sharey(ax10)
            ax[3,2].sharey(ax10)

            plt.show()
            # if not os.path.isfile(pred_image_dir+img[:-4]+".pdf"):
            #     plt.savefig(pred_image_dir+img[:-4]+".pdf")
            # plt.close()


    @staticmethod
    def _averaging(img, pool_size):
        raw_pred_rshp = img.reshape(1, img.shape[0], img.shape[1], 1)
        raw_pred_rshp = np.swapaxes(raw_pred_rshp, 3, 1)
        raw_pred_torch = torch.from_numpy(raw_pred_rshp)
        raw_pooled = F.avg_pool2d(input=raw_pred_torch,
                                  kernel_size=pool_size,
                                  stride=pool_size)
        # raw_out = F.upsample_bilinear(input=raw_pooled,
        #                               scale_factor=pool_size)
        raw_out = F.interpolate(input=raw_pooled, scale_factor=pool_size,
                                mode="bilinear")
        return np.transpose(raw_out.detach().numpy()[0][0])


if __name__ == "__main__":
    StridedConvAutoencoder.postprocessing(test_dir="/media/benjamin/blue_benny/data/test/")
