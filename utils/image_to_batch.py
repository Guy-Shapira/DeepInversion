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

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

teacher_bs_path = "deep_inverted_images/cifar100_n=10_resnet34/t5i/batch_"
copy_path = "/home/guy.shapira/DeepInv/DeepInversion/Deep/cifar100/deep_inverted_images/cifar100_n=10_resnet34/t5/"

def create_tesnors(num_batches, bs):
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


if __name__ == "__main__":
    create_tesnors(num_batches=170, bs=64)
