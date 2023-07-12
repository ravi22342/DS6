#!/usr/bin/env python
"""
Purpose: Predict Segmentation(s) for a set of 3D MRA volumes using pre-trained UNet or UNet-MSS network.
"""
import argparse
import random
import os
import numpy as np
import torch.utils.data
from pipeline import Pipeline
from Utils.model_manager import get_model

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-ds_path",
                        default="",
                        help="Path to folder containing dataset."
                             "Example: /home/dataset/")
    parser.add_argument("-out_path",
                        default="",
                        help="Folder path to store output "
                             "Example: /home/output/")
    parser.add_argument("-model",
                        type=int,
                        default=2,
                        help="1{U-Net}; \n"
                             "2{U-Net_Deepsup}; \n"
                             "3{Attention-U-Net}; \n"
                             "4{Probabilistic-U-Net};")
    parser.add_argument("-model_name",
                        default="Model_v1",
                        help="Name of the model")
    parser.add_argument('-train',
                        default=False,
                        help="To train the model")
    parser.add_argument('-test',
                        default=True,
                        help="To test the model")
    parser.add_argument('-deform',
                        default=False,
                        action="store_true",
                        help="To use deformation for training")
    parser.add_argument('-clip_grads',
                        default=True,
                        action="store_true",
                        help="To use deformation for training")
    parser.add_argument('-apex',
                        default=True,
                        help="To use half precision on model weights.")
    parser.add_argument("-batch_size",
                        type=int,
                        default=15,
                        help="Batch size for training")
    parser.add_argument("-num_epochs",
                        type=int,
                        default=50,
                        help="Number of epochs for training")
    parser.add_argument("-learning_rate",
                        type=float,
                        default=0.0001,
                        help="Learning rate")
    parser.add_argument("-patch_size",
                        type=int,
                        default=64,
                        help="Patch size of the input volume")
    parser.add_argument("-stride_depth",
                        type=int,
                        default=16,
                        help="Strides for dividing the input volume into patches in depth dimension "
                             "(To be used during validation and inference)")
    parser.add_argument("-stride_width",
                        type=int,
                        default=32,
                        help="Strides for dividing the input volume into patches in width dimension "
                             "(To be used during validation and inference)")
    parser.add_argument("-stride_length",
                        type=int,
                        default=32,
                        help="Strides for dividing the input volume into patches in length dimension "
                             "(To be used during validation and inference)")
    parser.add_argument("-samples_per_epoch",
                        type=int,
                        default=8000,
                        help="Number of samples per epoch")
    parser.add_argument("-num_worker",
                        type=int,
                        default=8,
                        help="Number of worker threads")
    parser.add_argument("-floss_coeff",
                        type=float,
                        default=0.7,
                        help="Loss coefficient for floss in total loss")
    parser.add_argument("-mip_loss_coeff",
                        type=float,
                        default=0.3,
                        help="Loss coefficient for mip_loss in total loss")
    parser.add_argument("-floss_param_smooth",
                        type=float,
                        default=1,
                        help="Loss coefficient for floss_param_smooth")
    parser.add_argument("-floss_param_gamma",
                        type=float,
                        default=0.75,
                        help="Loss coefficient for floss_param_gamma")
    parser.add_argument("-floss_param_alpha",
                        type=float,
                        default=0.7,
                        help="Loss coefficient for floss_param_alpha")
    parser.add_argument("-mip_loss_param_smooth",
                        type=float,
                        default=1,
                        help="Loss coefficient for mip_loss_param_smooth")
    parser.add_argument("-mip_loss_param_gamma",
                        type=float,
                        default=0.75,
                        help="Loss coefficient for mip_loss_param_gamma")
    parser.add_argument("-mip_loss_param_alpha",
                        type=float,
                        default=0.7,
                        help="Loss coefficient for mip_loss_param_alpha")
    parser.add_argument("-wandb",
                        default=True,
                        help="Set this to true to include wandb logging")

    args = parser.parse_args()

    if args.deform:
        args.model_name += "_Deform"

    MODEL_NAME = args.model_name
    DATASET_FOLDER = args.ds_path
    OUTPUT_PATH = args.out_path

    LOAD_PATH = os.path.realpath(__file__)
    LOAD_PATH = LOAD_PATH.replace("eval.py", "")
    CHECKPOINT_PATH = LOAD_PATH
    TENSORBOARD_PATH_TRAINING = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_training/'
    TENSORBOARD_PATH_VALIDATION = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_validation/'
    TENSORBOARD_PATH_TESTING = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_testing/'

    LOGGER_PATH = OUTPUT_PATH + "/" + MODEL_NAME + '.log'

    # Model
    model = torch.nn.DataParallel(get_model(args.model))
    model.cuda()

    pipeline = Pipeline(cmd_args=args, model=model, logger=None,
                        dir_path=DATASET_FOLDER, checkpoint_path=CHECKPOINT_PATH,
                        writer_training=None, writer_validating=None, wandb=False)
    pipeline.load(checkpoint_path=LOAD_PATH, load_best=True)
    pipeline.test(test_logger=None)

    torch.cuda.empty_cache()  # to avoid memory errors
