from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms as transforms

import numpy as np
import os
import glob
import collections

# from deepinversion_cifar10 import get_images
#
# try:
#     from apex.apex.parallel import DistributedDataParallel as DDP
#     from apex.apex import amp, optimizers
#     USE_APEX = True
# except ImportError:
#     print("Please install apex from https://www.github.com/nvidia/apex to run this example.")
#     print("will attempt to run without it")
#     USE_APEX = False

#provide intermeiate information
debug_output = False
debug_output = True

# device = 'cuda:6'
# CUDA_LAUNCH_BLOCKING=1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 DeepInversion')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--cig_scale', default=0.1, type=float, help='competition score')
    parser.add_argument('--di_lr', default=0.1, type=float, help='lr for deep inversion')
    parser.add_argument('--di_var_scale', default=0.001, type=float, help='TV L2 regularization coefficient')
    parser.add_argument('--di_l2_scale', default=0.0, type=float, help='L2 regularization coefficient')
    parser.add_argument('--r_feature_weight', default=10, type=float, help='weight for BN regularization statistic')
    parser.add_argument('--amp', action='store_true', help='use APEX AMP O1 acceleration')
    parser.add_argument('--exp_descr', default="try1", type=str, help='name to be added to experiment name')
    # parser.add_argument('--teacher_weights', default="'./checkpoint/teacher_resnet34_only.weights'", type=str, help='path to load weights of the model')
    parser.add_argument('--teacher_weights', default="pretrained/cifar100_n=10_resnet34/t5.pt", type=str, help='path to load weights of the model')
    parser.add_argument('--prefix', default="deep_inverted_images/cifar100_n=10_resnet34/t5", type=str, help='prefix of path to store pics ')
    args = parser.parse_args()


    for i in range(0, 171):
        cmd = 'python3.7 deepinversion.py --bs={} --r_feature_weight={} --di_lr={} --exp_descr={}' \
              ' --di_var_scale={} --di_l2_scale={} --cig_scale={} --counter={} --prefix={} --cuda=5'.format(args.bs, args.r_feature_weight,
                                                                                       args.di_lr, args.exp_descr,
                                                                                       args.di_var_scale,
                                                                                       args.di_l2_scale,
                                                                                       args.cig_scale, i, args.prefix)
        os.system(cmd)
