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


import matplotlib
import matplotlib.pyplot as plt

class NetTrainer():
    def __init__(self, net, criterion, optimizer, starting_max_acc=0):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer

        self.max_val_acc = starting_max_acc

    def ed(self, ds, indices, howmany):
        tots = len(indices)
        self.net.eval()
        list_of_errors = []

        randomized_list = numpy.random.choice([x for x in indices], len(indices), replace=False)

        dataloaders = [tud.DataLoader(ds._train_val_set, batch_size=100, shuffle=False,
             num_workers=2, sampler=customcifar.CustomSampler(randomized_list)) for i in range(2)]

        with torch.no_grad():
            for batch_index, element in enumerate(zip(*dataloaders)):
                els = [x for x in element]

                for input in els:
                    input[0], input[1] = input[0].to("cuda:0"), input[1].to("cuda:0")
                outputs = [self.net(i[0]) for i in els]

                confidence = acquisition_functions.entropic_distance(outputs)
                for x in range(len(confidence)):
                    list_of_errors.append([confidence[x], els[0][2][x].item()])
                print("\r Checked: {0} / {1}".format(len(list_of_errors), tots), end='')
            sorlist = sorted(list_of_errors, key=lambda xp: xp[0], reverse=True)

            return [el[1] for el in sorlist[:howmany]]

    def entropy(self, ds, indices, howmany):
        tots = len(indices)
        self.net.eval()
        list_of_errors = []

        randomized_list = numpy.random.choice([x for x in indices], len(indices), replace=False)

        dataloaders = [tud.DataLoader(ds._train_val_set, batch_size=100, shuffle=False,
             num_workers=2, sampler=customcifar.CustomSampler(randomized_list)) for i in range(5)]

        with torch.no_grad():
            for batch_index, element in enumerate(zip(*dataloaders)):
                els = [x for x in element]

                for input in els:
                    input[0], input[1] = input[0].to("cuda:0"), input[1].to("cuda:0")
                outputs = [self.net(i[0]) for i in els]

                confidence = acquisition_functions.avg_entropy(outputs)
                for x in range(len(confidence)):
                    list_of_errors.append([confidence[x], els[0][2][x].item()])
                #  print(list_of_errors)
                print("\r Checked: {0} / {1}".format(len(list_of_errors), tots), end='')
            sorlist = sorted(list_of_errors, key=lambda xp: xp[0], reverse=True)

            return [el[1] for el in sorlist[:howmany]]

    def distance_and_varratio(self, ds, indices, howmany, train_indices, n=5):
        distance_weight = 1
        varratio_weight = 0

        self.net.eval()
        N = torch.Tensor().to("cuda:0")  # labelled
        S = torch.Tensor().to("cuda:0")  # unlabelled
        normalized_confidence = [torch.Tensor().to("cuda:0"), torch.Tensor().long()]

        randomized_list = numpy.random.choice([x for x in indices], len(indices), replace=False)

        trainloaders = [tud.DataLoader(ds._train_val_set, batch_size=500, shuffle=False, num_workers=4,
                                       sampler=customcifar.CustomRandomSampler(train_indices)) for i in range(n)]
        dataloaders = [tud.DataLoader(ds._train_val_set, batch_size=500, shuffle=False, num_workers=4,
                                      sampler=customcifar.CustomSampler(randomized_list)) for i in range(n)]
        with torch.no_grad():
            for batch_index, element in enumerate(zip(*trainloaders)):  # labelled samples
                els = [x for x in element]
                o = torch.Tensor().to("cuda:0")
                for input in els:
                    input[0], input[1] = input[0].to("cuda:0"), input[1].to("cuda:0")
                    o = torch.cat((o, self.net(input[0])[1].reshape(len(input[0]), 512, 1)), 2)
                N = torch.cat((N, o), 0)
                print("\r N: {0} ".format(N.size()), end="")
            print("")

            for batch_index, element in enumerate(zip(*dataloaders)):  # unlabelled samples
                normalized_confidence[1] = torch.cat((normalized_confidence[1], element[0][2]), 0)

                els = [x for x in element]
                o = torch.Tensor().to("cuda:0")
                predictions = torch.Tensor().long()

                for input in els:
                    input[0], input[1] = input[0].to("cuda:0"), input[1].to("cuda:0")
                    output = self.net(input[0])
                    out = output[1].reshape(len(input[0]), 512, 1)

                    o = torch.cat((o, out), 2)
                    predictions = torch.cat((predictions, output[0].max(1)[1].reshape(len(output[0]), 1).cpu()), 1)

                normalized_confidence[0] = torch.cat((normalized_confidence[0].cpu(), 1 - torch.Tensor(
                    acquisition_functions.confidence(predictions.transpose(0,1))).cpu() / n), 0).cpu()
                print(normalized_confidence)

                S = torch.cat((S, o), 0)
                print("\r S: {0} ".format(S.size()), end="")
            print("")
            S = (torch.sum(S, 2)) / n
            N = (torch.sum(N, 2)) / n

            S_batches = torch.split(S, 25, dim =0)
            dist_S_N = torch.Tensor()
            for el in S_batches:
                partial_dist = el.unsqueeze(1) - N.unsqueeze(0)
                partial_dist = torch.sum(partial_dist * partial_dist, -1)
                partial_dist = torch.sqrt(partial_dist)
                dist_S_N = torch.cat((dist_S_N, partial_dist.cpu()), 0)

            mindist = torch.min(dist_S_N, 1)[0].to("cuda:0")

            normalizing_factor = torch.max(mindist, -1)[0]
            print("NF : " + str(normalizing_factor))

            mindist_confidence =  (mindist / normalizing_factor) + normalized_confidence[0].to("cuda:0") # devo calcolare la confidenza ancora

            erlist_indexes = normalized_confidence[1]
            new_N = []

            for i in range(howmany):
                #  maxx = torch.max(mindist, -1)[1]
                maxx = torch.max(mindist_confidence, -1)[1]
                print("Max: {0:.3f} = {1:.3f} + {2:.3f}".format(mindist_confidence[maxx], mindist[maxx]/normalizing_factor, normalized_confidence[0][maxx]))

                if erlist_indexes[maxx].item() in new_N:
                    print("Error: Duplicate")

                new_N.append(erlist_indexes[maxx].item())
                mindist[maxx] = float("-inf")
                mindist_confidence[maxx] = float("-inf")

                newdists = S - S[maxx].reshape(1, len(S[maxx]))
                newdists = torch.sum(newdists * newdists, -1)
                newdists = torch.sqrt(newdists)
                mindist = torch.min(mindist, newdists)
                mindist_confidence = (distance_weight*(mindist / normalizing_factor)) + (varratio_weight * normalized_confidence[0].to("cuda:0"))
            return new_N

    def kl_divergence(self, ds, indices, howmany, train_indices, n=5):
        self.net.eval()
        N = torch.Tensor().to("cuda:0") #labelled
        S = torch.Tensor().to("cuda:0") #unlabelled
        normalized_confidence = [torch.Tensor().to("cuda:0"), torch.Tensor().long()]


        randomized_list = numpy.random.choice([x for x in indices], len(indices), replace=False)

        trainloaders = [tud.DataLoader(ds._train_val_set, batch_size=500, shuffle=False, num_workers=4,
                                    sampler=customcifar.CustomRandomSampler(train_indices)) for i in range(n)]
        dataloaders = [tud.DataLoader(ds._train_val_set, batch_size=500, shuffle=False, num_workers=4,
                                      sampler=customcifar.CustomSampler(randomized_list)) for i in range(n)]
        with torch.no_grad():
            for batch_index, element in enumerate(zip(*trainloaders)): #labelled samples
                els = [x for x in element]
                o = torch.Tensor().to("cuda:0")
                for input in els:
                    input[0], input[1] = input[0].to("cuda:0"), input[1].to("cuda:0")
                    o = torch.cat((o, self.net(input[0])[0].reshape(len(input[0]),10, 1)), 2)
                N = torch.cat((N, o), 0)
                print("\r N: {0} ".format(N.size()), end="")
            print("")

            for batch_index, element in enumerate(zip(*dataloaders)): #unlabelled samples
                normalized_confidence[1] = torch.cat((normalized_confidence[1], element[0][2]), 0)

                els = [x for x in element]
                o = torch.Tensor().to("cuda:0")
                predictions = torch.Tensor().long().to("cuda:0")
                for input in els:
                    input[0], input[1] = input[0].to("cuda:0"), input[1].to("cuda:0")
                    out = self.net(input[0])[0].reshape(len(input[0]), 10, 1)
                    o = torch.cat((o, out), 2)
                    predictions = torch.cat((predictions, out.max(1)[1]), 1).to("cuda:0")
                normalized_confidence[0] = torch.cat((normalized_confidence[0].cpu(), 1.1 - torch.Tensor(acquisition_functions.confidence(predictions.transpose(1, 0))).cpu() / n), 0).cpu()

                S = torch.cat((S, o), 0)
                print("\r S: {0} ".format(S.size()), end="")
            print("")

            # calc KL divergence
            S = (torch.sum(F.softmax(S, dim=1), 2)) /n
            N = (torch.sum(F.softmax(N, dim=1), 2)) /n

            S_on_N = S.to("cpu").unsqueeze(1) / N.to("cpu").unsqueeze(0)
            ln_S_on_N = numpy.log2(S_on_N).reshape(len(N), len(S), 10).transpose(0,1)

            ln_S_on_N_batches = torch.split(ln_S_on_N, 300, dim=0)
            S_batches = torch.split(S, 300, dim=0)




            kldiv = torch.Tensor()
            for i in range(len(ln_S_on_N_batches)):
                partial_kldiv = torch.bmm(ln_S_on_N_batches[i].to("cuda:0"), S_batches[i].reshape(len(S_batches[i]), 10, 1)).cpu()
                kldiv = torch.cat((partial_kldiv, kldiv), 0)
                print(kldiv.size())
            kldiv = kldiv.reshape(len(S), len(N))

            mindiv = torch.min(kldiv, 1)[0]* normalized_confidence[0]
            errorlist = [[mindiv[i].item(), normalized_confidence[1][i].item() ]for i in range(len(normalized_confidence[0]))]
            sorlist = sorted(errorlist, key=lambda xp: xp[0], reverse=True)

            return [x[1] for x in sorlist[:howmany]]


    def greedy_k_centers(self, ds, indices, howmany, _train_loader, n=5):

        self.kl_divergence(ds, indices, howmany, _train_loader)

        self.net.eval()
        N = torch.Tensor().to("cuda:0")
        S = torch.Tensor().to("cuda:0")

        randomized_list = numpy.random.choice([x for x in indices], len(indices), replace=False)

        dataloader = tud.DataLoader(ds._train_val_set, batch_size=500, shuffle=False, num_workers=2,
                                      sampler=customcifar.CustomSampler(randomized_list))
        with torch.no_grad():
            for batch_index, (inputs, targets, index) in enumerate(_train_loader):
                inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
                outputs = self.net(inputs)[0]
                N = torch.cat((N, outputs), 0)

            for batch_index, (inputs, targets, index) in enumerate(dataloader):
                # x = arg max(i in S/N) min(j in N) d(X_i, X_j)
                inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
                outputs = self.net(inputs)[0]
                S = torch.cat((S, outputs), 0)

        differences = S.to("cpu").unsqueeze(1) - N.to("cpu").unsqueeze(0)
        print(differences.size())
        dist_m = torch.sum(differences * differences, -1).pow(.5)

        mindist = [x for x in zip(randomized_list, torch.min(dist_m.to("cuda:0"), 1)[0].to("cpu").data)]
        sorlist = sorted(mindist, key=lambda xp: xp[1].item(), reverse=True)
        print(sorlist)
        return [x[0] for x in sorlist[:howmany]]

    def bestofn(self, ds, indices, howmany, n=5):
        self.net.eval()
        total_normalized_confidence = 0
        total = 0
        list_of_errors = []

        errors_by_class = [0 for x in range(10)]
        printiter = 0

        randomized_list = numpy.random.choice([x for x in indices], len(indices), replace=False)
        dataloaders = [tud.DataLoader(ds._train_val_set, batch_size=500, shuffle=False,
             num_workers=2, sampler=customcifar.CustomSampler(randomized_list)) for i in range(n)]

        with torch.no_grad():
            for batch_index, element in enumerate(zip(*dataloaders)):
                els = [x for x in element]
                for input in els:
                    input[0], input[1] = input[0].to("cuda:0"), input[1].to("cuda:0")
                res_net = [self.net(i[0]) for i in els]

                outputs = [i[0] for i in res_net]
                intrep = [i[1] for i in res_net][0]

                # intrep = torch.cat((intrep, [i[1] for i in res_net][0]), 0)
                predictions = [out.max(1)[1] for out in outputs]

                normalized_confidence = [float(c/n) for c in acquisition_functions.confidence(predictions)]
                differences = intrep.unsqueeze(1) - intrep.unsqueeze(0)
                dist_m = torch.sum(differences * differences, -1).pow(.5)
                for x in range(len(normalized_confidence)):
                    sbregio = [dist_m[x][y].item() for y in range(len(dist_m[x])) if y!=x]
                    mindist = min(sbregio)

                    list_of_errors.append([(1 -normalized_confidence[x]) * mindist, els[0][2][x].item()])
                    if normalized_confidence[x] < 1:
                        errors_by_class[els[0][1][x].item()] += 1
                    total_normalized_confidence += normalized_confidence[x]
                    total += 1
                    if printiter % 50 == 0:
                        print("\r Avg confidence: {0:.2f}% ({1:.1f}/{2})  {3}".format((total_normalized_confidence / total)*100, total_normalized_confidence, total, ""), end='')
                    printiter += 1

            # qui va cambiato
            sorlist = sorted(list_of_errors, key=lambda xp: xp[0], reverse=True)
            print("\n Errors by class:  {0}".format(["{0}: {1}".format(i, errors_by_class[i]) for i in range(10)]))
            return [el[1] for el in sorlist[:howmany]]


    def train(self, epoch, _train_loader):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0

        printiter = 0
        b_i = 0

        # debug
        elementsperclass = [0] * 10

        for batch_index, (inputs, targets, index) in enumerate(_train_loader):
            inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")

            self.optimizer.zero_grad()
            outputs = self.net(inputs)[0]
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


            # debug
            for el in targets:
                elementsperclass[el.item()] += 1

            if printiter % 5 == 0 :
                print("\r{0:<60} {1} ".format("\t Loss: {0:.3f} | Acc: {1:.3f} ({2}/{3}) ".format(train_loss / (batch_index + 1), 100. * correct / total, correct, total), elementsperclass), end='')

            printiter+= 1

        print("\r{0:<60} {1} ".format(
            "\t Loss: {0:.3f} | Acc: {1:.3f} ({2}/{3}) ".format(train_loss / (b_i + 1), 100. * correct / total,
                                                                correct, total), elementsperclass))

    def test(self, epoch, _test_loader, filename='res'):
        self.net.eval()
        test_loss = 0
        correct = 0

        accuracy_per_class = [[0,0] for x in range(10)]

        total = 0
        printiter = 0
        b_i = 0
        with torch.no_grad():
            for batch_index, (inputs, targets, index) in enumerate(_test_loader):
                inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
                outputs = self.net(inputs)[0]
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                for e1, e2 in zip(predicted, targets):
                    if e1.item() == e2.item():
                        accuracy_per_class[e1.item()][0] += 1
                    accuracy_per_class[e2.item()][1] += 1

                if printiter % 5 == 0:
                    print('\r Test: Loss: %.3f | Acc: %.3f%% (%d/%d) ' % (
                    test_loss / (batch_index + 1), 100. * correct / total, correct, total), end='')
                printiter += 1
                b_i = batch_index


        print('\r Test: Loss: %.3f | Acc: %.3f%% (%d/%d) ' % (
            test_loss / (b_i + 1), 100. * correct / total, correct, total))

        if filename is not None:
            with open(filename + "_per_class.csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([accuracy_per_class[i][0] for i in range(len(accuracy_per_class))])

        for el in range(len(accuracy_per_class)):
            print("CLASS: {0} - Acc: {1:.3f} ({2}/{3})".format(el, accuracy_per_class[el][0] / accuracy_per_class[el][1], accuracy_per_class[el][0], accuracy_per_class[el][1]))
        return 100. * correct / total

    def clone(self):
        return NetTrainer(net=copy.deepcopy(self.net), optimizer=self.optimizer, criterion=self.criterion, starting_max_acc=self.max_val_acc)

    def validate(self, epoch, _validation_loader):
        self.net.eval()
        validation_loss = 0
        correct = 0
        total = 0
        printiter= 0
        b_i = 0

        elementsperclass = [0] * 10

        with torch.no_grad():
            for batch_index, (inputs, targets, index) in enumerate(_validation_loader):
                inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
                outputs = self.net(inputs)[0]
                loss = self.criterion(outputs, targets)

                validation_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                for el in targets:
                    elementsperclass[el.item()] += 1

                if printiter % 5 == 0:
                    print("\r{0:<60} {1} ".format('\tValidation: Loss: %.3f | Acc: %.3f%% (%d/%d) ' % (
                        validation_loss / (batch_index + 1), 100. * correct / total, correct, total),elementsperclass ), end='')
                printiter += 1
                b_i += 1
        print("\r{0:<60} {1} ".format('\tValidation: Loss: %.3f | Acc: %.3f%% (%d/%d) ' % (
            validation_loss / (b_i + 1), 100. * correct / total, correct, total), elementsperclass))


        if (100. * correct / total) > self.max_val_acc :
            print("Accuracy is improved! (from: {0:.2f})".format(self.max_val_acc))
            self.max_val_acc = 100. * correct / total
            return True, self.max_val_acc
        else:
            print("Max Acc still: {0:.2f}".format(self.max_val_acc))
            return False, 100. * correct / total

    def reset_acc(self):
        self.max_val_acc = 0
        return self
