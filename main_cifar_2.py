import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud

import torchvision.transforms as trans

import numpy
import copy
import csv
import datetime

import prova_torch_resnet as netter
import customcifar
import net_functions as nf
import utils

num_of_classes = 10
val_percentage = .2
initial_percentage = .3
learning_rate = 0.001
num_of_epochs = 100



transform = trans.Compose([
        # trans.RandomRotation(5),
        # trans.RandomCrop(26),
        # trans.Resize((32, 32)),
        # utils.Gauss(0, 0.05),
        trans.ToTensor()
    ])


class CompleteDataset:
    def __init__(self):
        self.dataset = customcifar.CustomCIFAR10(root="./cifar", train=True, download=True, transform=transform)
        self.testset = customcifar.CustomCIFAR10(root="./cifar", train=False, download=True, transform=transform) # palindromo!

        dataloader = tud.DataLoader(self.dataset, batch_size=64, shuffle=False, num_workers=2,
                                    sampler=customcifar.CustomRandomSampler([x for x in range(len(self.dataset))]))
        el_for_class = [[] for x in range(num_of_classes)]
        for batch_index, (inputs, targets, index) in enumerate(dataloader):
            for t in range(len(targets)):
                el_for_class[targets[t]].append(index[t].item())

        val_els_per_class = int((len(self.dataset) * val_percentage) / num_of_classes)

        self.validation_indices = [el for xl in el_for_class for el in numpy.random.choice(xl, size=val_els_per_class, replace=False)]
        self.remaining_indices = [x for x in range(len(self.dataset)) if x not in self.validation_indices]
        self.train_indices = numpy.random.choice(self.remaining_indices, size=int(len(self.remaining_indices)*initial_percentage  ), replace=False)

        print("Dataset loaded: train length {0}/{3} | validation length {1} | test length {2}".format(len(self.train_indices), len(self.validation_indices), len(self.testset), len(self.remaining_indices)))


def new_net():
    net = netter.CustomResNet18()
    net = net.to("cuda:0")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    return nf.NetTrainer(net=net, criterion=criterion, optimizer=optimizer)




cd = CompleteDataset()

# dataloader = tud.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, sampler=customcifar.CustomRandomSampler(indices))
for i in range(num_of_epochs):
    net = new_net()
    best_net = net.clone()

#     net.train(i, dataloader)

#    isbest, acc = net.validate(i, dataset.validate())
    # print("Accuracy so far: {0:.2f}".format(acc))
#    if isbest:
    best_network = net.clone()

print("\n\t  TEST:")
# best_acc = net.test(0, dataset.test())
# print("Test accuracy: {0:.2f}".format(best_acc))





