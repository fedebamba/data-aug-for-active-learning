import copy

import torch
import torch.utils.data as tud
import torch.nn.functional as F

import utils
import csv
import numpy
import time

import customcifar
import acquisition_functions




def generate_weak_labels(net, cds, indices, howmany, train_indices, n=5):
    net.eval()

    normalized_confidence = [torch.Tensor().to("cuda:0"), torch.Tensor().long()]

    randomized_list = numpy.random.choice([x for x in indices], len(indices), replace=False)
    dataloaders = [tud.DataLoader(cds.train_indices, batch_size=500, shuffle=False, num_workers=4,
                                  sampler=customcifar.CustomSampler(randomized_list)) for i in range(n)]

    with torch.no_grad():
        for batch_index, element in enumerate(zip(*dataloaders)):  # unlabelled samples
            normalized_confidence[1] = torch.cat((normalized_confidence[1], element[0][2]), 0)

            els = [x for x in element]
            o = torch.Tensor().to("cuda:0")
            predictions = torch.Tensor().long()

            for input in els:
                input[0], input[1] = input[0].to("cuda:0"), input[1].to("cuda:0")
                output = net(input[0])
#                out = output[1].reshape(len(input[0]), 512, 1)

#                o = torch.cat((o, out), 2)
                predictions = torch.cat((predictions, output[0].max(1)[1].reshape(len(output[0]), 1).cpu()), 1)

            print(predictions)
            normalized_confidence[0] = torch.cat((normalized_confidence[0].cpu(), 1 - torch.Tensor(
                acquisition_functions.confidence(predictions.transpose(0,1), details=True)).cpu() / n), 0).cpu()

            print(normalized_confidence)
            
