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
                      out_channels=1024,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.LeakyReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=1024,
                      out_channels=1024,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.LeakyReLU()
        )


        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=1024,
                      out_channels=1024,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
            nn.LeakyReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=1024,
                      out_channels=1024,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
            nn.LeakyReLU()
        )

        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,
                               out_channels=1024,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.LeakyReLU(),
        )

        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,
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
    def train_model(model, train_data, n_epochs=5, img_size=4096): ################<---------------------
        start = time.time()
        model.float()
        model_dir = "../models/"
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_filepath = model_dir+"ae_"+str(img_size)+"_gpu.pth"
        if os.path.isfile(model_filepath):
            model = torch.load(model_filepath)
            print("trained model already found")
            model = model.cuda()
            return model

        already_completed_epochs = n_epochs
        temp_model_filepath = model_dir+"ae_"+str(img_size)+"_epoch_"+\
                              str(already_completed_epochs)+"_gpu.pth"
        while not os.path.isfile(temp_model_filepath) and \
                  already_completed_epochs > 0:
            already_completed_epochs -= 1
            temp_model_filepath = model_dir+"ae_"+str(img_size)+"_epoch_"+\
                                  str(already_completed_epochs)+"_gpu.pth"

        if already_completed_epochs>0:
            del model
            torch.cuda.empty_cache()
            model = torch.load(temp_model_filepath)

        model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=[0,1,2]).cuda()

        # set loss and optimizer
        criterion = nn.MSELoss().cuda()
        optimizer = optim.Adam(model.parameters())

        # create torch data loader
        train_loader = data.DataLoader(train_data, batch_size=1, shuffle=True,
                                       num_workers=1, pin_memory=True)

        min_train_loss = 1000000.

        # train model
        print("training model...")
        for epoch in range(already_completed_epochs+1, n_epochs+1):

            epoch_start = time.time()
            loss = 0.0
            for (noisy, clean) in train_loader:
                noisy = noisy.cuda()
                clean = clean.cuda()

                # clear the gradients of all optimized variables
                for param in model.parameters():
                    param.grad = None

                # forward pass
                outputs = model(noisy)
                del noisy

                # calculate the loss
                instance_loss = criterion(outputs, clean)
                del clean

                # backprop
                instance_loss.backward()

                # update parameters
                optimizer.step()

                # update training loss
                loss += instance_loss.item()
                del instance_loss

                torch.cuda.empty_cache()

            # print avg training loss
            loss /= len(train_loader)
            print("Epoch: "+str(epoch)+", Training Loss:"+str(loss),
                  "Time: "+str((time.time()-epoch_start)/60)+"min")

            # if epoch < n_epochs:
            #     torch.save(model, model_dir+"ae_"+str(img_size)+"_epoch_"+\
            #                       str(epoch)+"_gpu.pth")
            # prior_filepath = model_dir+"ae_"+str(img_size)+"_epoch_"+\
            #                  str(epoch-1)+"_gpu.pth"
            # if epoch > 1 and os.path.isfile(prior_filepath):
            #     os.remove(prior_filepath)

        # torch.save(model, model_dir+"ae_"+str(img_size)+"_gpu.pth")

        return model


    @staticmethod
    def validate_on_train_img(model, eval_data, img_size=4096):
        print("--- validation predictions ---")
        eval_loader = data.DataLoader(eval_data, batch_size=1, shuffle=False,
                                      num_workers = 1, pin_memory=True)

        predictions_dir = "../predictions/eval_"+str(img_size)+"/"
        if not os.path.isdir(predictions_dir):
            os.mkdir(predictions_dir)
        pred_image_dir = predictions_dir+"/images/"
        if not os.path.isdir(pred_image_dir):
            os.mkdir(pred_image_dir)

        model.eval()
        with torch.no_grad():
            for i, img in enumerate(eval_loader):
                raw, _ = img
                raw = raw.cuda()

                prediction_raw_filename = predictions_dir+\
                                          eval_data.img_list[i][:-4]+\
                                          "_pred_raw.npy"
                if os.path.isfile(prediction_raw_filename):
                    continue
                pred = model(raw).cpu()
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
    def make_predictions(model, test_data, img_size=4096):

        test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False,
                                      num_workers = 1, pin_memory=True)

        predictions_dir = "../predictions/test_"+str(img_size)+"/"
        if not os.path.isdir(predictions_dir):
            os.mkdir(predictions_dir)

        model.eval()
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                raw_pred_filename = predictions_dir+\
                                    test_data.img_list[i][:-4]+\
                                    "_pred_raw.npy"
                denoise_pred_filename = predictions_dir+\
                                        test_data.img_list[i][:-4]+\
                                        "_pred_denoise.npy"
                if os.path.isfile(raw_pred_filename) and \
                   os.path.isfile(denoise_pred_filename):
                    continue

                raw, denoised = img

                raw_prediction = model(raw)
                raw_prediction = raw_prediction.detach().numpy()[0][0]
                raw_prediction = np.transpose(raw_prediction)
                np.save(raw_pred_filename, raw_prediction)
                del raw

                denoise_prediction = model(denoised)
                denoise_prediction = denoise_prediction.detach().numpy()[0][0]
                denoise_prediction = np.transpose(denoise_prediction)
                np.save(denoise_pred_filename, denoise_prediction)
                del denoised
                torch.cuda.empty_cache()

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

            nansen = np.load(test_data.img_dir+"denoised/"+img[:-4]+".npy")
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


class StridedConvAutoencoderParallel(nn.Module):
    def __init__(self):
        print("parallel model")
        super().__init__()

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
                      out_channels=1024,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.LeakyReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=1024,
                      out_channels=1024,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.LeakyReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=1024,
                      out_channels=1024,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
            nn.LeakyReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=1024,
                      out_channels=1024,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
            nn.LeakyReLU()
        )

        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,
                               out_channels=1024,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.LeakyReLU(),
        )

        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,
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

        self.conv1.cuda(0)
        self.conv2.cuda(1)
        self.conv3.cuda(2)
        self.conv4.cuda(0)
        self.conv5.cuda(1)
        self.conv6.cuda(2)
        self.conv7.cuda(0)
        self.conv8.cuda(1)
        self.tconv6.cuda(0)
        self.tconv5.cuda(1)
        self.tconv4.cuda(2)
        self.tconv3.cuda(0)
        self.tconv2.cuda(2)
        self.tconv1.cuda(1)
        self.final.cuda(0)


    def forward(self, x):
        x = x.cuda(0)
        conv1_out = self.conv1(x)

        conv1_out = conv1_out.cuda(1)
        conv2_out = self.conv2(conv1_out)

        conv2_out = conv2_out.cuda(2)
        conv3_out = self.conv3(conv2_out)

        conv3_out = conv3_out.cuda(0)
        conv4_out = self.conv4(conv3_out)

        conv4_out = conv4_out.cuda(1)
        conv5_out = self.conv5(conv4_out)

        conv5_out = conv5_out.cuda(2)
        conv6_out = self.conv6(conv5_out)

        conv6_out = conv6_out.cuda(0)
        conv7_out = self.conv7(conv6_out)

        conv7_out = conv7_out.cuda(1)
        conv8_out = self.conv8(conv7_out)

        conv8_out = conv8_out.cuda(0)
        tconv6_out = self.tconv6(conv8_out)

        tconv6_out = tconv6_out.cuda(1)
        tconv5_out = self.tconv5(tconv6_out)

        tconv5_out = tconv5_out.cuda(2)
        tconv4_out = self.tconv4(tconv5_out)

        tconv4_out = tconv4_out.cuda(0)
        tconv4_and_skip = torch.cat([tconv4_out, conv3_out], 1)
        tconv3_out = self.tconv3(tconv4_and_skip)

        tconv3_out = tconv3_out.cuda(2)
        tconv3_and_skip = torch.cat([tconv3_out, conv2_out], 1)
        tconv2_out = self.tconv2(tconv3_and_skip)

        tconv2_out = tconv2_out.cuda(1)
        tconv2_and_skip = torch.cat([tconv2_out, conv1_out], 1)
        tconv1_out = self.tconv1(tconv2_and_skip)

        tconv1_out = tconv1_out.cuda(0)
        tconv1_and_skip = torch.cat([tconv1_out, x], 1)
        out = self.final(tconv1_and_skip)

        return out



    @staticmethod
    def train_model(model, train_data, n_epochs=5, img_size=4096): ################<---------------------
        start = time.time()
        model.float()
        model_dir = "../models/"
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_filepath = model_dir+"ae_"+str(img_size)+"_gpu.pth"
        if os.path.isfile(model_filepath):
            model = torch.load(model_filepath)
            print("trained model already found")
            # model = model.cuda()
            return model

        already_completed_epochs = n_epochs
        temp_model_filepath = model_dir+"ae_"+str(img_size)+"_epoch_"+\
                              str(already_completed_epochs)+"_gpu.pth"
        while not os.path.isfile(temp_model_filepath) and \
                  already_completed_epochs > 0:
            already_completed_epochs -= 1
            temp_model_filepath = model_dir+"ae_"+str(img_size)+"_epoch_"+\
                                  str(already_completed_epochs)+"_gpu.pth"

        if already_completed_epochs>0:
            del model
            torch.cuda.empty_cache()
            model = torch.load(temp_model_filepath)

        # model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=[0,1,2]).cuda()

        # set loss and optimizer
        criterion = nn.MSELoss().cuda()
        optimizer = optim.Adam(model.parameters())

        # create torch data loader
        train_loader = data.DataLoader(train_data, batch_size=1, shuffle=True,
                                       num_workers=1, pin_memory=True)

        min_train_loss = 1000000.

        # train model
        print("training model...")
        for epoch in range(already_completed_epochs+1, n_epochs+1):

            epoch_start = time.time()
            loss = 0.0
            for (noisy, clean) in train_loader:
                noisy = noisy.cuda()
                clean = clean.cuda()

                # clear the gradients of all optimized variables
                for param in model.parameters():
                    param.grad = None

                # forward pass
                outputs = model(noisy)
                del noisy

                # calculate the loss
                instance_loss = criterion(outputs, clean)
                del clean

                # backprop
                instance_loss.backward()

                # update parameters
                optimizer.step()

                # update training loss
                loss += instance_loss.item()
                del instance_loss

                torch.cuda.empty_cache()

            # print avg training loss
            loss /= len(train_loader)
            print("Epoch: "+str(epoch)+", Training Loss:"+str(loss),
                  "Time: "+str((time.time()-epoch_start)/60)+"min")

            # if epoch < n_epochs:
            #     torch.save(model, model_dir+"ae_"+str(img_size)+"_epoch_"+\
            #                       str(epoch)+"_gpu.pth")
            # prior_filepath = model_dir+"ae_"+str(img_size)+"_epoch_"+\
            #                  str(epoch-1)+"_gpu.pth"
            # if epoch > 1 and os.path.isfile(prior_filepath):
            #     os.remove(prior_filepath)

        # torch.save(model, model_dir+"ae_"+str(img_size)+"_gpu.pth")

        return model


    @staticmethod
    def validate_on_train_img(model, eval_data, img_size=4096):
        print("--- validation predictions ---")
        eval_loader = data.DataLoader(eval_data, batch_size=1, shuffle=False,
                                      num_workers = 1, pin_memory=True)

        predictions_dir = "../predictions/eval_"+str(img_size)+"/"
        if not os.path.isdir(predictions_dir):
            os.mkdir(predictions_dir)
        pred_image_dir = predictions_dir+"/images/"
        if not os.path.isdir(pred_image_dir):
            os.mkdir(pred_image_dir)

        model.eval()
        with torch.no_grad():
            for i, img in enumerate(eval_loader):
                raw, _ = img
                raw = raw.cuda()

                prediction_raw_filename = predictions_dir+\
                                          eval_data.img_list[i][:-4]+\
                                          "_pred_raw.npy"
                if os.path.isfile(prediction_raw_filename):
                    continue
                pred = model(raw).cpu()
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
    def make_predictions(model, test_data, img_size=4096):

        test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False,
                                      num_workers = 1, pin_memory=True)

        predictions_dir = "../predictions/test_"+str(img_size)+"/"
        if not os.path.isdir(predictions_dir):
            os.mkdir(predictions_dir)

        model.eval()
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                raw_pred_filename = predictions_dir+\
                                    test_data.img_list[i][:-4]+\
                                    "_pred_raw.npy"
                denoise_pred_filename = predictions_dir+\
                                        test_data.img_list[i][:-4]+\
                                        "_pred_denoise.npy"
                if os.path.isfile(raw_pred_filename) and \
                   os.path.isfile(denoise_pred_filename):
                    continue

                raw, denoised = img

                raw_prediction = model(raw)
                raw_prediction = raw_prediction.detach().numpy()[0][0]
                raw_prediction = np.transpose(raw_prediction)
                np.save(raw_pred_filename, raw_prediction)
                del raw

                denoise_prediction = model(denoised)
                denoise_prediction = denoise_prediction.detach().numpy()[0][0]
                denoise_prediction = np.transpose(denoise_prediction)
                np.save(denoise_pred_filename, denoise_prediction)
                del denoised
                torch.cuda.empty_cache()

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

            nansen = np.load(test_data.img_dir+"denoised/"+img[:-4]+".npy")
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
