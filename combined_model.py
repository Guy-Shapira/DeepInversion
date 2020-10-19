


import torch
import torchvision
import numpy as np
import torch.nn as nn
import tqdm
import sys
import os
from torchvision.transforms import ToTensor
from ResNetCifar import ResNet18, ResNet34, ResNet50
from utils import get_split_cifar100
import argparse

def onehot(num_classes, y, device):
    y_hot = torch.zeros((len(y), num_classes)).to(device)
    for i, label in enumerate(y):
        y_hot[i][label] = 1

    return y_hot


def student_train(student_net, teachers, num_epochs, bs, num_batches, data_dir, save_path, num_classes, device):

    optimizer = torch.optim.SGD(student_net.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0002)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 15, 25, 40], gamma=0.5)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    for epoch in range(num_epochs):
        with tqdm.tqdm(total=num_batches, file=sys.stdout) as pbar:
            losses = [[] for _ in range(len(teachers))]
            accuracy = [[] for _ in range(len(teachers))]
            # load batch from each teacher, add to loss and then backward and step
            for i in reversed(range(0, num_batches)):
                loss = None
                dscr_str = ""
                optimizer.zero_grad()
                for tidx, teacher in enumerate(teachers):
                    teacher = teacher.to(device)
                    batch = torch.load(data_dir + "/t" + str(tidx + 1) + "/batch" + str(i) + ".pt")

                    batch = batch.to(device)
                    teacher = teacher.to(device)
                    teacher_labels = teacher(batch)
                    teacher_labels = torch.argmax(teacher_labels, dim=1).to(device)
                    teacher_labels += (tidx) * num_classes
                    pred = student_net(batch).to(device)
                    curr_loss = loss_fn(pred, teacher_labels.long())
                    if loss is None:
                        loss = curr_loss
                    else:
                        loss += curr_loss
                    losses[tidx].append(curr_loss.item())

                    teacher_labels = onehot(len(teachers) * num_classes, teacher_labels, device=device).to(device)
                    acc = (pred.argmax(-1) == teacher_labels.argmax(-1)).sum().item()
                    acc /= bs
                    accuracy[tidx].append(acc)
                    dscr_str += '_{}_loss = {}, _{}_acc = {} '.format(tidx + 1, curr_loss.item(), tidx + 1, acc)
                    del(teacher)


                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_net.parameters(), 0.5)
                optimizer.step()

                pbar.set_description(dscr_str)
                pbar.update()

        epoch_str = ""
        for tidx in range(len(teachers)):
            epoch_str += 'teacher_{}_loss = {}, teacher_{}_acc = {} \n'.format(tidx + 1, sum(losses[tidx]) / len(losses[tidx]), tidx + 1, sum(accuracy[tidx]) / len(accuracy[tidx]))
        scheduler.step()
        epoch_str += 'epoch = {}'.format(epoch)
        print(epoch_str)
        pbar.update()

        if epoch % 1 == 0:
            # Checking student accuracy
            print("Checking student accuracy")
            test(student_net=student_net, len_teachers=len(teachers), num_classes=num_classes, bs=bs, device=device)
            student_net = student_net.train()
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(student_net.state_dict(), save_path + "/checkpoint_" + str(epoch) + ".pt")


    torch.save(student_net.state_dict(), save_path + "/final.pt")


def test(student_net, len_teachers, num_classes, bs, device):
    print('Started student validation')
    student_net.eval()
    losses = [0 for _ in range(len_teachers)]
    correct = [0 for _ in range(len_teachers)]
    total = [0 for _ in range(len_teachers)]
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        batch_idx = 0
        for i in range(len_teachers):
            _, testloader = get_split_cifar100(bs, num_classes * i, num_classes * (i + 1))

            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student_net(inputs)
                loss = criterion(outputs, targets)

                losses[i] += loss.item()
                _, predicted = outputs.topk(5, 1, True, True)
                predicted = predicted.t()
                correct[i] += predicted.eq(targets.view(1, -1).expand_as(predicted)).sum().item()



                # _, predicted = outputs.max(1)
                total[i] += targets.size(0)
                # correct[i] += predicted.eq(targets).sum().item()
                batch_idx += 1

            print('Teacher {}:'.format(i + 1))
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (losses[i] / (1 * (batch_idx + 1)), 100. * correct[i] / total[i], correct[i], total[i]))
    print('Total:')
    print('Acc: %.3f%% (%d/%d)'
        % (100. * sum(correct) / sum(total), sum(correct), sum(total)))
    print('Finished student validation')



def main():
    parser = argparse.ArgumentParser(description='Combining teachers to student')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--num_batches', default=64, type=int, help='number of batches for each teacher')
    parser.add_argument('--n', default=2, type=int, help='number of teachers')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes per teacher')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs to train')
    parser.add_argument('--teacher_arch', default='ResNet34', type=str, help='teachers architecture')
    parser.add_argument('--checkpoint_dir', type=str, help='path in which pretrain teacher checkpoints are stored. Format is (t1.pt, t2.pt,...,tn.pt) in dir')
    parser.add_argument('--data_dir', type=str, help= """path in which deep inverted data is stored. Expects to find dirs t1, t2, t3, ..., tn. In each t_i
                        folder shold be all the deep inverted data as batch1.pt, batch2.pt, ...""")
    parser.add_argument('--save_path', type=str, help='path to store student model in')
    args = parser.parse_args()
    num_teachers = args.n
    num_classes = args.num_classes
    teachers_list = []
    device = 'cuda:7'
    print('Loading teacher from checkpoints')
    for i in range(1, num_teachers + 1):
        if(args.teacher_arch == 'ResNet34'):
            new_teacher = ResNet34(num_classes)
        else:
            new_teacher = ResNet50(num_classes)
        new_teacher.load_state_dict(torch.load(args.checkpoint_dir + "/t{}".format(i) + ".pt"))
        new_teacher  = new_teacher.eval()
        teachers_list.append(new_teacher)

    print('Creating student model')
    student_net = ResNet18(num_teachers * num_classes).to(device)
    print('Start training')
    student_train(student_net=student_net, teachers=teachers_list, num_epochs=args.num_epochs, num_batches=args.num_batches, data_dir=args.data_dir,
    save_path=args.save_path, num_classes=args.num_classes, device=device, bs=args.bs)


if __name__ == "__main__":
    main()
