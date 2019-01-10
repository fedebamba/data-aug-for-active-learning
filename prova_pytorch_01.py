import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud

import torchvision.transforms as trans

import numpy
import copy
import csv
import time
import datetime

import prova_torch_resnet as netter
import customcifar
import net_functions as nf


if __name__ == '__main__':



    # parameters...........................................
    learning_rate = 5e-5

    percentage = 3
    train_set_percentage = 12

    print([x for x in iter(([1,3,5,8,12,345,56576,4]))])




    print("   {0}".format({"ciao" : 1}))

    with open("ciao.csv", "w+") as file:
        fieldnames = ["a", "b", "c"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writerow({"a": "porco dio"})
        writer.writeheader()
        writer.writerow({"a":1, "b":2, "c": 3})



    epochs_first_step = 10  # 50
    epochs_second_step = 10

    train_batch_size = 32

    train_set_length = 50000  # total length of training set data
    train_set_percentage = 10
    tslp = int((train_set_length * train_set_percentage) / 100)

    excluded_training_indices = []

    # data loading and stuff...............................
    traintrans_01 = trans.Compose([
        trans.ToTensor()
    ])

    traintrans_02 = trans.Compose([
        # trans.RandomCrop(28),
        trans.RandomCrop(24),
        trans.RandomRotation(5),
        # trans.RandomHorizontalFlip(.5),
        trans.Resize((32, 32)),
        trans.ToTensor()
    ])
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # network definition........................................

    net = netter.ResNet18()
    net = net.to("cuda:0")

    criterion = nn.CrossEntropyLoss()  # probabilmente la dovremo cambiare
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    net_trainer = nf.NetTrainer(net=net, criterion=criterion, optimizer=optimizer)

    # training & test set.......................................
    train_set = customcifar.CustomCIFAR10(root="./cifar", train=True, download=True, transform=traintrans_01)
    transformed_train_set = customcifar.CustomCIFAR10(root="./cifar", train=True, download=True,
                                                      transform=traintrans_02)
    test_set = customcifar.CustomCIFAR10(root="./cifar", train=False, download=True, transform=traintrans_01)  # 10000

    bal_training_set = net_trainer.select_class_balanced_trainingset(train_set)[1]
    train_loader = tud.DataLoader(train_set, batch_size=train_batch_size, shuffle=False, num_workers=2,
                                  sampler=tud.SequentialSampler([x for x in range(5000)]))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    # main.........................................
    best_acc = 0

    for i in range(epochs_first_step):
        print("\t====================\n\t  TRAIN:  " + str(i) + "\n\t====================")
        net_trainer.train(i, train_loader)
        print("\t====================\n\t  TEST:   " + str(i) + "\n\t====================")
        acc = net_trainer.test(i, test_loader)
        if acc > best_acc:
            best_acc = acc
            print("Best Accuracy so far: " + str(best_acc))




    random = []
    loe = []
    for percentage in range(1, int(100 / train_set_percentage)):
        # create the 2 datasets and dataloaders
        print("Train percentage: {0:.1f}".format((percentage + 1)*train_set_percentage))

        # eligible indexes for active learning
        random_indices = numpy.random.choice([x for x in range(train_set_length) if x not in bal_training_set and x not in loe], train_set_length - (tslp + percentage), False)

        train_loader_01 = tud.DataLoader(train_set, batch_size=train_batch_size, shuffle=False, num_workers=2, sampler=tud.SequentialSampler(random_indices))
        train_loader_02 = tud.DataLoader(transformed_train_set, batch_size=train_batch_size, shuffle=False, num_workers=2, sampler=tud.SequentialSampler(random_indices))

        # random elements for the normal-trained network
        random = numpy.random.choice([x for x in range(train_set_length) if x not in bal_training_set and x not in random], tslp, False)

        # list of indexes for active learning
        loe = net_trainer.select_new_trainset(train_loader_01, train_loader_02, tslp)
        if len(loe) < tslp:
            # in case there aren't enough errors....
            loe = loe.extend([x for x in random if x not in loe and len(loe) <= tslp])

        print("r_i: {0} | random: {1} | loe: {2}".format(len(random_indices), len(random), len(loe)))
        train_loader_normal = tud.DataLoader(train_set, batch_size=train_batch_size, shuffle=False, num_workers=2, sampler=tud.SubsetRandomSampler(numpy.append(bal_training_set, random)))
        train_loader_active = tud.DataLoader(train_set, batch_size=train_batch_size, shuffle=False, num_workers=2, sampler=tud.SubsetRandomSampler(numpy.append(bal_training_set, loe)))

        # create another network for comparison
        other_net_trainer = net_trainer.clone()

        results = []

        # train the network for a specified number of epochs
        for i in range(epochs_second_step):
            print("\t====================\n\t  TRAIN NET ACTIVE :  " + str(i) + "\n\t====================")
            net_trainer.train(i, train_loader_active)
            print("\t====================\n\t  TRAIN NET NORMAL :  " + str(i) + "\n\t====================")
            other_net_trainer.train(i, train_loader_normal)

            print("\t====================\n\t  TEST NET ACTIVE :   " + str(i) + "\n\t====================")
            acc_active = net_trainer.test(i, test_loader)
            print("\t====================\n\t  TEST NET NORMAL :   " + str(i) + "\n\t====================")
            acc_normal = other_net_trainer.test(i, test_loader)

            print("Normal acc.: {0}% / Active acc.: {1}% || d:{2:.2f}%".format(acc_normal, acc_active, acc_active - acc_normal))
            results.append({'epoch':i, 'delta': (acc_active - acc_normal), 'active':acc_active, 'normal':acc_normal})

    print("RES: ")
    for i in results:
        print("\t > {0:.2f}".format(i))

    with open("results.csv", "w+") as csvfile:
        fieldnames = ["epoch","active","normal","delta"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
