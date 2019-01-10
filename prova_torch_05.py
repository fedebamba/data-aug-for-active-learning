import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as trans

import matplotlib.pyplot as mpl
import numpy
import time

import prova_torch_resnet as netter
import utils
import customcifar


if __name__ == '__main__':

    # parameters.......................
    learning_rate = 5e-5

    # data loading.....................
    # What i'm trying to do it's using different transformations of the same dataset (cifar) to obtain different
    # augmented versions of the same image; then i'll have to figure how to feed these images to the network (because i
    # think that just using the dataloader batches won't work)

    traintrans_01 = trans.Compose([
        trans.ToTensor()
    ])
    traintrans_02 = trans.Compose([
        trans.CenterCrop(20),
        trans.Resize((32, 32)),
        trans.ToTensor()
    ])

    # train_01 = torchvision.datasets.CIFAR10(root="./cifar", train=True, download=True, transform=traintrans_01)
    # train_02 = torchvision.datasets.CIFAR10(root="./cifar", train=True, download=True, transform=traintrans_02)

    train_01 = customcifar.CustomCIFAR10(root="./cifar", train=True, download=True, transform=traintrans_01, percentage=0.15)
    train_02 = customcifar.CustomCIFAR10(root="./cifar", train=True, download=True, transform=traintrans_02)

    trainloader_01 = torch.utils.data.DataLoader(train_01, batch_size=4, shuffle=False, num_workers=2)
    trainloader_02 = torch.utils.data.DataLoader(train_02, batch_size=4, shuffle=False, num_workers=2)

    # SHOW AN IMAGE...........
    #utils.show_image(124, train_01, train_02)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # network definition...............
    net = netter.ResNet18()
    net = net.to("cuda:0")

    criterion = nn.CrossEntropyLoss()  # probabilmente la dovremo cambiare
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)


    def select_new_trainset(epoch):
        data_aug_correct = 0
        list_of_errors = []

        for batch_index, (input1, t1, i1),(input2, t2, i2) in enumerate(zip(trainloader_01, trainloader_02)):
            input1, t1 = input1.to("cuda:0"), t1.to("cuda:0")
            input2, t2 = input2.to("cuda:0"), t2.to("cuda:0")

            out1 = net(input1)
            x, pred1 = out1.max(1)
            out2 = net(input2)
            x, pred2 = out2.max(1)

            for (el1, el2, i) in zip(pred1, pred2, i1):
                if el1 == el2:
                    data_aug_correct += 1
                else:
                    list_of_errors.extend(i)

    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        data_aug_correct = 0

        for batch_index, ((inputs1, targets1, index1), (inputs2, targets2, index2)) in enumerate(zip(trainloader_01, trainloader_02)):

            inputs1, targets1 = inputs1.to("cuda:0"), targets1.to("cuda:0")
            inputs2, targets2 = inputs2.to("cuda:0"), targets2.to("cuda:0")

            outputs2 = net(inputs2)
            x, pred2 = outputs2.max(1)

            optimizer.zero_grad()

            outputs1 = net(inputs1)
            x, pred1 = outputs1.max(1)

            for (el1, el2, i) in zip(pred1, pred2, index1):
                if el1 == el2:
                    data_aug_correct += 1
                else:
                    print("\t>>> " + str(i))

            loss = criterion(outputs1, targets1)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs1.max(1)

            total += targets1.size(0)

            correct += predicted.eq(targets1).sum().item()

            print('Loss: %.3f | Acc: %.3f%% (%d/%d) | DataAug errs: %d'% (train_loss / (batch_index + 1), 100. * correct / total, correct, total, total - data_aug_correct))
    for i in range(300):
        train(i)
