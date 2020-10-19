from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms as transforms

import numpy as np
import os
import glob
import collections

#provide intermeiate information
debug_output = False
debug_output = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 DeepInversion')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--bn', default=64, type=int, help='number_of_batches')
    parser.add_argument('--cig_scale', default=0.1, type=float, help='competition score')
    parser.add_argument('--di_lr', default=0.1, type=float, help='lr for deep inversion')
    parser.add_argument('--di_var_scale', default=0.001, type=float, help='TV L2 regularization coefficient')
    parser.add_argument('--di_l2_scale', default=0.0, type=float, help='L2 regularization coefficient')
    parser.add_argument('--r_feature_weight', default=10, type=float, help='weight for BN regularization statistic')
    parser.add_argument('--amp', action='store_true', help='use APEX AMP O1 acceleration')
    parser.add_argument('--exp_descr', default="try1", type=str, help='name to be added to experiment name')
    parser.add_argument('--teacher_weights', default="pretrained/cifar100_n=10_resnet34/t5.pt", type=str, help='path to load weights of the model')
    parser.add_argument('--prefix', default="deep_inverted_images/cifar100_n=10_resnet34/t5", type=str, help='prefix of path to store pics ')
    parser.add_argument('--cuda', type=int, help='cuda to run on')
    parser.add_argument('--max_label', type=int, help='number of classes to create images from')
    args = parser.parse_args()


    for i in range(0, args.bn):
        cmd = 'python3.7 deepinversion.py --bs={} --r_feature_weight={} --di_lr={} --exp_descr={}' \
              ' --di_var_scale={} --di_l2_scale={} --cig_scale={} --counter={} --prefix={} --cuda={} --max_label={}'.format(args.bs, args.r_feature_weight,
                                                                                       args.di_lr, args.exp_descr,
                                                                                       args.di_var_scale,
                                                                                       args.di_l2_scale,
                                                                                       args.cig_scale, i, args.prefix, args.cuda, args.max_label)
        os.system(cmd)
