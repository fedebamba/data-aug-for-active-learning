
import torchvision
import torch.utils.data as tud

import math
import numpy
import csv

class CustomCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, indices=None, percentage=0.0, other=False):
        super().__init__(root=root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)


    def __getitem__(self, index):
        (img, target) = super().__getitem__(index)
        return img, target, index

    def __len__(self):
        return super().__len__()

    def __repr__(self):
        return super().__repr__()



class UnbalancedCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, provided_indices=None, num_full_classes=5, percentage=.1, valels=200, filename=None):
        super().__init__(root=root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if train:
            if provided_indices is not None:
                self._val_indices = provided_indices[1]
                self.indices = provided_indices[0]
                self.el_for_class = None
                # self.train_data = self.train_data[self.indices]
            else:
                full_classes = numpy.random.choice([x for x in range(10)], size=num_full_classes,replace=False)
                el_for_class = [[] for x in range(10)]
                data_loader = tud.DataLoader(self, batch_size=100, shuffle=False, num_workers=2,
                                             sampler=CustomSampler([x for x in range(len(self.train_data))]))

                for batch_index, (input, target, i) in enumerate(data_loader):
                    for x in range(len(input)):
                        el_for_class[target[x].item()].append(i[x].item())

                for i in range(len(el_for_class)):
                    if i not in full_classes:
                        el_for_class[i] = el_for_class[i][:int(len(el_for_class[i])*percentage)]

                print(["{0}:{1}".format(i, len(el_for_class[i])) for i in range(10)])

                self._val_indices = [x for el in el_for_class for x in numpy.random.choice(el, valels, False)]
                self.indices = [x for el in el_for_class for x in el]
                self.el_for_class = el_for_class

                # self.train_data = self.train_data[self.indices]
                if filename is not None:
                    with open(filename + "_per_class.csv", "w") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["100 %" if x in full_classes else "{0} %".format(int(percentage * 100)) for x in range(10)])


            print('Train data ' + str(len(self.train_data)))

    def __getitem__(self, index):
        (img, target) = super().__getitem__(index)
        return img, target, index


    def __len__(self):
        return super().__len__()

    def __repr__(self):
        return super().__repr__()


class CustomSampler(tud.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)


class CustomRandomSampler(tud.Sampler):
    def __init__(self, data_source):
        arr = numpy.random.choice(data_source, len(data_source), False)
        self.data_source = arr

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)

    def refresh(self):
        self.data_source = numpy.random.choice(self.data_source, len(self.data_source), False)