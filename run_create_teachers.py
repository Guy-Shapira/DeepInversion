from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR100/n DeepInversion')
    parser.add_argument('--bs', default=1024, type=int, help='batch size')
    parser.add_argument('--n', default=10, type=str, help='number of parts to divide in')
    parser.add_argument('--resnet',default=34,type=int,help='Resnet architecture')
    parser.add_argument('--epochs',default=120,type=int,help='num epochs to train')
    args = parser.parse_args()

    n = int(args.n)
    num_classes = int(100/n)
    for i in range(1, n):
        start_class = i*num_classes
        end_class = (i+1)*num_classes
        cmd = 'python3 partial_cifar_train_teacher.py --bs={} --n={} --start_class={} --end_class={} --resnet={} --num_epochs={}'.format(args.bs,args.n,start_class,end_class,args.resnet,args.epochs)
        os.system(cmd)
