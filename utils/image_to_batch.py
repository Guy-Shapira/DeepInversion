import torch
import torchvision
import numpy as np
import torch.nn as nn
import tqdm
import sys
import os
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import argparse

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])


def create_tesnors(num_batches, bs, teacher_bs_path, copy_path):
    for i in range(num_batches):
        batch = None
        if not os.path.isfile(teacher_bs_path + str(i) + "/tensor.pt"):
            for j in range(0, bs):
                img = Image.open(teacher_bs_path + str(i) + "/best_images/output_0" + "_" + str(j + 1) + ".png")
                tensor_img = torch.unsqueeze(transform(img), dim=0)
                if batch is None:
                    batch = tensor_img
                else:
                    batch = torch.cat((batch, tensor_img), dim=0)
                    file_path = teacher_bs_path + str(i) + "/batch" + str(i) + ".pt"
        torch.save(batch, file_path)
        cmd = 'sudo scp ' + file_path + " " + copy_path
        os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description='Combining teachers to student')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--num_batches', default=64, type=int, help='number of batches for each teacher')
    parser.add_argument('--teacher_bs_path' help='path in which images are stored')
    parser.add_argument('--copy_path', help='path to store images in')
    args = parser.parse_args()
    create_tesnors(num_batches=args.num_batches, bs=args.bs,
        teacher_bs_path=agrs.teacher_bs_path, copy_path=args.copy_path)


if __name__ == "__main__":
    main()
