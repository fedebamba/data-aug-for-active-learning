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



# PARAMETER PART................

esname = "exp_umb_" + str(datetime.datetime.now().strftime("%B.%d.%Y-%H.%M"))
just100 = False

learning_rate = 0.0001

epochs_first_step = 50  # 50
epochs_second_step = 50

train_batch_size = 32

total_train_data = 27500
train_val_ratio = .9
train_set_percentage = 5

first_time_multiplier = 1
until_slice_number = 8

train_set_length = int(total_train_data-2000)  # total length of training set data
tslp = int((train_set_length * train_set_percentage) / 100)


traintrans_01 = trans.Compose([
        trans.RandomRotation(5),
        trans.RandomCrop(26),
        trans.Resize((32, 32)),
        utils.Gauss(0, 0.02),
        trans.ToTensor()
    ])
traintrans_02 = trans.Compose([
    # trans.RandomCrop(28),
    trans.RandomCrop(24),
    trans.RandomRotation(5),
    # utils.Gauss(0, 0.02),
    # trans.RandomHorizontalFlip(.5),
    trans.Resize((32, 32)),
    trans.ToTensor()
])

class CifarLoader():
    def __init__(self, transform=None, first_time_multiplier=1, name=None, joking=False):
        if joking:
            return

        self._train_val_set = customcifar.UnbalancedCIFAR10(root="./cifar", train=True, download=True, transform=transform, filename=name, percentage=.1)


        self._test_set = customcifar.UnbalancedCIFAR10(root="./cifar", train=False, download=True, transform=transform)  # 10000


        self.validation_indices = self._train_val_set._val_indices
        self.train_indices = [x for x in self._train_val_set.indices if x not in self.validation_indices]
        self.already_selected_indices = numpy.random.choice(self.train_indices, size=tslp*first_time_multiplier, replace=False).tolist()
        self._train = tud.DataLoader(self._train_val_set, batch_size=train_batch_size, shuffle=False, num_workers=2,
                                  sampler=customcifar.CustomRandomSampler(self.already_selected_indices))

        self._v = tud.DataLoader(self._train_val_set, batch_size=100, shuffle=False, num_workers=2,
                                          sampler=customcifar.CustomRandomSampler(self.validation_indices))
        self._t = torch.utils.data.DataLoader(self._test_set, batch_size=100, shuffle=False, num_workers=2, sampler=customcifar.CustomSampler([x for x in range(len((self._test_set)))]))

    def restore(self, all, selected, validation, transform=None, name=None):
        self._train_val_set = customcifar.UnbalancedCIFAR10(root="./cifar", train=True, download=True,
                                                            transform=transform, filename=name, percentage=.1, provided_indices=(all, validation))
        self._test_set = customcifar.UnbalancedCIFAR10(root="./cifar", train=False, download=True,
                                                       transform=transform)  # 10000
        self.validation_indices = validation

        self.train_indices = [x for x in all if x not in self.validation_indices]
        self.already_selected_indices=selected
        self._train = tud.DataLoader(self._train_val_set, batch_size=train_batch_size, shuffle=False, num_workers=2,
                                     sampler=customcifar.CustomRandomSampler(self.already_selected_indices))
        self._v = tud.DataLoader(self._train_val_set, batch_size=100, shuffle=False, num_workers=2,
                                 sampler=customcifar.CustomRandomSampler(self.validation_indices))
        self._t = torch.utils.data.DataLoader(self._test_set, batch_size=100, shuffle=False, num_workers=2,
                                              sampler=customcifar.CustomSampler(
                                                  [x for x in range(len((self._test_set)))]))
        return self



    def all_train(self, otherDS=None, excluded=[]):
        if otherDS is None:
            return tud.DataLoader(self._train_val_set, batch_size=1, shuffle=False,
                                         num_workers=2, sampler=customcifar.CustomSampler([x for x in self.train_indices if x not in excluded]))
        else:
            return tud.DataLoader(otherDS, batch_size=1, shuffle=False,
                                         num_workers=2, sampler=customcifar.CustomSampler([x for x in self.train_indices if x not in excluded]))

    def train(self):
        return self._train

    def validate(self):
        return tud.DataLoader(self._train_val_set, batch_size=100, shuffle=False, num_workers=2,
                                    sampler=customcifar.CustomRandomSampler(self.validation_indices))
    def test(self):
        return  torch.utils.data.DataLoader(self._test_set, batch_size=100, shuffle=False, num_workers=2)

    def select_for_train(self, indices):
        self.already_selected_indices.extend(indices)
        return tud.DataLoader(self._train_val_set, batch_size=train_batch_size, shuffle=False, num_workers=2,
                                    sampler=customcifar.CustomRandomSampler(indices))


# ////////////////////////////////////////////
def new_network():
    # net = netter.ResNet18()
    net = netter.CustomResNet18()
    net = net.to("cuda:0")

    criterion = nn.CrossEntropyLoss()  # probabilmente la dovremo cambiare
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    return nf.NetTrainer(net=net, criterion=criterion, optimizer=optimizer)


def write_dataset_info(ds, active_indices, normal_indices, filename):
    active_els = [0] * 10
    normal_els = [0] * 10

    dataloader_1 = tud.DataLoader(ds._train_val_set, batch_size=1, shuffle=False,
                                  num_workers=2, sampler=customcifar.CustomSampler([x for x in active_indices]))
    dataloader_2 = tud.DataLoader(ds._train_val_set, batch_size=1, shuffle=False,
                                  num_workers=2, sampler=customcifar.CustomSampler([x for x in normal_indices]))


    print([len([el for el in active_indices if ds._train_val_set.train_labels[el] == i]) for i in range(10)])


    with torch.no_grad():
        for b, (input, target, i) in enumerate(dataloader_1):
            active_els[target.item()] += 1
        for b, (input, target, i) in enumerate(dataloader_2):
            normal_els[target.item()] += 1

    with open(filename + "_datainfo.csv", "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([active_els[i] for i in range(len(active_els))] + [""] + [normal_els[i] for i in range(len(normal_els))])





def single_train_batch(num_of_epochs=10, dataset=None, indices=None, name=None):
    network = new_network()
    best_network = network.clone()

    for i in range(num_of_epochs):
        print("\n\t  TRAIN:  " + str(i))
        if indices is None:
            network.train(i, dataset.train())
        else:
            network.train(i, dataset.select_for_train(indices))
        print("\t  VALIDATION:   " + str(i))
        isbest, acc = network.validate(i, dataset.validate())
        if isbest:
            best_network = network.clone()
        else:
            pass
            # network = best_network.clone()

    print("\n\t  TEST:")
    best_acc = network.test(0, dataset.test(), name)
    print("Test accuracy: {0:.2f}".format(best_acc))

    return best_network, best_acc

def save_the_net(thename):
    net_trainer = new_network()

    # Dataset def
    dataset = CifarLoader(transform=traintrans_01, first_time_multiplier=first_time_multiplier,
                          name=None)
    best_net, best_acc = single_train_batch(num_of_epochs=epochs_first_step, dataset=dataset,
                                            name=None)

    torch.save(best_net.net, "models/" + thename + ".pt")
    with open("models/" + thename + "_indexes.csv", "w+" ) as file:
        writer = csv.writer(file)
        writer.writerow(x for x in dataset._train_val_set.indices)
        writer.writerow([x for x in dataset.already_selected_indices])
        writer.writerow([x for x in dataset.validation_indices])

def load_the_net(thename):
    model = torch.load("models/" + thename + ".pt")
    model.eval()

    with open("models/" + thename + "_indexes.csv", "r+" ) as file:
        reader = csv.reader(file)
        all_indices = next(reader)
        all_indices = [int(x) for x in all_indices]

        selected_indices = next(reader)
        selected_indices = [int(x) for x in selected_indices]

        validation_indices = next(reader)
        validation_indices = [int(x) for x in validation_indices]

        return model, all_indices, selected_indices, validation_indices
# MAIN.....................................................

have_to_save_it = False
thename = "starting_net"
if have_to_save_it:
    save_the_net(thename=thename)

net, b, c, d = load_the_net(thename=thename)
net = net.to("cuda:0")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

net_trainer = nf.NetTrainer(net=net, criterion=criterion, optimizer=optimizer)

dataset = CifarLoader(transform=None, first_time_multiplier=first_time_multiplier,
                          name=None, joking=True).restore(b, c, d, transform=traintrans_01, name=None)
el_for_active = [x for x in dataset.already_selected_indices]
# active_indices = net_trainer.bestofn(dataset, [x for x in dataset.train_indices if x not in el_for_active], tslp)
net_trainer.distance_and_varratio(dataset, [x for x in dataset.train_indices if x not in el_for_active], tslp, el_for_active)
# net_trainer.greedy_k_centers(dataset, [x for x in dataset.train_indices if x not in el_for_active], tslp, dataset.select_for_train(el_for_active))