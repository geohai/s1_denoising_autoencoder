import os
import sys
import argparse
import torch

# from s1_dataset import Sentinel1
from s1_dataset_numpy import Sentinel1np
from conv_autoencoder import StridedConvAutoencoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="../data/",
                        help="directory where downloaded images are living")
    parser.add_argument("-a", "--aug", type=bool, default=False,
                        help="whether to use data augmentation in training \
                              images during training")
    return parser.parse_args()


def main():
    args = parse_args()

    train_data = Sentinel1np(image_dir=args.dir, dataset="train")
    print(1/0)
    train_data.save_training_img()

    model = StridedConvAutoencoder()
    model = StridedConvAutoencoder.train_model(model=model,
                                               train_data=train_data)
    del train_data
    print(1/0)

    eval_data = Sentinel1np(image_dir=args.dir, dataset="eval")
    StridedConvAutoencoder.validate_on_train_img(model=model,
                                                 eval_data=eval_data)
    del eval_data

    test_data = Sentinel1np(image_dir=args.dir, dataset="test")
    StridedConvAutoencoder.make_predictions(model=model,
                                            test_data=test_data)
    # StridedConvAutoencoder.postprocessing(test_dir=test_data.img_dir)
    del test_data

if __name__ == '__main__':
    main()
