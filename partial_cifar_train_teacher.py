# 2019.07.24-Changed output of forward function
# Huawei Technologies Co., Ltd. <foss@huawei.com>
# taken from https://github.com/huawei-noah/Data-Efficient-Model-Compression/blob/master/DAFL/resnet.py
# for comparison with DAFL


import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.drop1 = nn.Dropout(p=0.55)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.drop2 = nn.Dropout(p=0.55)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.drop2(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_feature=False):
        x = self.conv1(x)

        x = self.bn1(x)
        out = F.relu(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out, feature


def ResNet18(num_classes=50):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=50):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=50):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=50):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=100):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)





import torch
import torchvision
import numpy as np
import torch.nn as nn
import tqdm
import sys
import os

def get_split_cifar100( batch_size=32, start=0, end=50):
    shuffle = False
    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    # start_class = (task_id - 1) * 5
    start_class = start
    # end_class = task_id * 5
    end_class = end

    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=transform_train)
    test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=transform_test)

    targets_train = torch.tensor(train.targets)
    target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))

    targets_test = torch.tensor(test.targets)
    target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(train, np.where(target_train_idx == 1)[0]), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(test, np.where(target_test_idx == 1)[0]),
                                              batch_size=batch_size)

    return train_loader, test_loader

def onehot(num_classes, y):
    y_hot = torch.zeros((len(y), num_classes)).to(device)
    for i,label in enumerate(y):
        y_hot[i][label] = 1

    return y_hot


def train(model, num_epochs, bs, device = "cuda:1", start_class=0,end_class=50,dir_name = "./cifar100_def",num_classes = 50):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    train_loader, test_loader = get_split_cifar100(bs, start_class, end_class)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.085, momentum=0.85, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 90, 200], gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    model = model.to(device)


    for epoch in range(num_epochs):
        total_loss_train, total_loss_test = 0, 0
        total_acc_train, total_acc_test = 0, 0
        for data in train_loader:
            x, y = data
            y-=start_class
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x).to(device)
            loss = loss_fn(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.65)
            optimizer.step()
            y = onehot(num_classes,y).to(device)

            acc = (pred.argmax(-1) == y.argmax(-1)).sum().item()
            acc /= bs
            total_loss_train += loss
            total_acc_train += (pred.argmax(-1) == y.argmax(-1)).sum().item()

        total_samples= len(train_loader)*bs
        total_loss_train /= len(train_loader)
        total_acc_train /= total_samples

        scheduler.step()


        if epoch%10 == 0:
            save_path = str(dir_name) +"/" + str(epoch) + ".pt"
            torch.save(model.state_dict(), save_path)

            with torch.no_grad():
                for data in test_loader:
                    x, y = data
                    x = x.to(device)
                    y -= start_class
                    y = y.to(device)
                    pred = model(x).to(device)
                    loss = loss_fn(pred, y)
                    y = onehot(num_classes,y).to(device)

                    acc = (pred.argmax(-1) == y.argmax(-1)).sum().item()
                    acc /= bs
                    total_loss_test += loss
                    total_acc_test += (pred.argmax(-1) == y.argmax(-1)).sum().item()

                total_samples = len(test_loader) * bs
                total_loss_test /= len(test_loader)
                total_acc_test /= total_samples
                print(f'Training: avg batch loss={total_loss_train:4.5f}, total accuracy={total_acc_train:3.5f} Testing: avg batch loss={total_loss_test:4.5f}, total accuracy={total_acc_test:3.5f} epoch={epoch}')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = str(save_dir)+"/sgd_p2_final.pt"
    torch.save(model.state_dict(), save_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 DeepInversion')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--n', default=2, type=int, help='classes partition size')
    parser.add_argument('--start_class', default=0, type=int, help='start class (included)')
    parser.add_argument('--end_class', default=50, type=int, help='end_class (not included)')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs to train')
    parser.add_argument('--resnet', default=34,type=int,help='Resnet type of teacher model')
    args = parser.parse_args()

    print("training Resnet"+str(args.resnet)+" strat_class="+str(args.start_class)+" end_class="+str(args.end_class))


    device = "cuda:7"
    num_classes = int(100/int(args.n))
    if int(args.resnet) == 34:
        model = ResNet34(num_classes)
    else:
        model = ResNet50(num_classes)

    if not os.path.exists("./resnet"+str(args.resnet)+"split_cifar_to" + str(args.n)):
        os.mkdir("./resnet"+str(args.resnet)+"split_cifar_to" + str(args.n))

    save_dir = "./resnet"+str(args.resnet)+"split_cifar_to" + str(args.n) + "/start:"+str(args.start_class)+"_end:"+str(args.end_class)
    train(model, args.num_epochs, args.bs, device,args.start_class,args.end_class,dir_name=save_dir,num_classes=num_classes)
