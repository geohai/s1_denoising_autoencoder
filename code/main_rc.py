import os
import sys
import argparse
import torch

from s1_dataset_rc import Sentinel1
from conv_autoencoder import StridedConvAutoencoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="../data/",
                        help="directory where downloaded images are living")
    return parser.parse_args()


def main():
    args = parse_args()
    train_data = Sentinel1(image_dir=args.dir, dataset="train")
    # train_data.save_training_img()
    # print("Training images saved")

    model = StridedConvAutoencoder()
    model = StridedConvAutoencoder.train_model(model=model,
                                               train_data=train_data)
    del train_data

    eval_data = Sentinel1(image_dir=args.dir, dataset="eval")
    StridedConvAutoencoder.validate_on_train_img(model=model,
                                                 eval_data=eval_data)
    del eval_data

    test_data = Sentinel1(image_dir=args.dir, dataset="test")
    StridedConvAutoencoder.make_predictions(model=model,
                                            test_data=test_data)
    del test_data

if __name__ == '__main__':
    main()
