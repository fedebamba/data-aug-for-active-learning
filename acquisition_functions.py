import torch.nn.functional as F
import torch as T

import numpy


def entropy(net_out):
    sm = F.softmax(net_out)
    logsm = numpy.log2(sm)
    return -numpy.multiply(sm.cpu().numpy(), logsm.cpu().numpy()).sum()


def avg_entropy(net_out):
    e = numpy.zeros(shape=len(net_out[0]))
    for el in net_out:
        sm = F.softmax(el)
        logsm = numpy.log2(sm)
        f = T.mul(T.tensor(sm).to("cuda:0"), T.tensor(logsm).to("cuda:0")).double()
        e = e - T.sum(f.cpu(), 1)
    return e/len(net_out)


def confidence(vector, num_of_classes=10):
    res =  [max([(len([i[v] for i in vector if i[v] == j])) for j in range(num_of_classes)]) for v in range(len(vector[0]))]
    return res

def avg_KL_divergence():
    pass




def max_variance(net_out):
    # it's the same as bestofn
    pred = [out.max(1)[1] for out in net_out]
    return 1 - (confidence(pred)/len(net_out))


    #   e = numpy.zeros(shape=(len(net_out[0]), 10))
    # for el in net_out:
    #    sm = F.softmax(el)
    #    e = T.dist(e, sm, 1)
    #    print(e)

    # return e.max(1)


def entropic_distance(net_out):
    sm = F.softmax(net_out[0])
    logsm = numpy.log2(sm)
    e = - T.mul(T.tensor(sm).to("cuda:0"), T.tensor(logsm).to("cuda:0")).double().cpu()

    sm = F.softmax(net_out[1])
    logsm = numpy.log2(sm)
    f = - T.mul(T.tensor(sm).to("cuda:0"), T.tensor(logsm).to("cuda:0")).double().cpu()

    ee = T.sum((e - f).abs(), 1)
    return ee
