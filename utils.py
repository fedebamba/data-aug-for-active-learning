import matplotlib.pyplot as mpl
import csv
import numpy
from PIL import Image

def show(f1,f2):
    fig = mpl.figure(figsize=(8,8))

    fig.add_subplot(1, 2, 1)
    mpl.imshow(f1[0][1, :])

    fig.add_subplot(1, 2, 2)
    mpl.imshow(f2[0][1, :])
    mpl.show()


def gen_gnp(mean=0, var=1):
    _m = mean
    _v = var
    def gaussian_noise_pass(x):
        r, c = x.size
        g = numpy.random.normal(_m, _v ** .5, (r, c, 3)).reshape(r, c, 3)

        imray = numpy.array(x.getdata())
        i = numpy.uint8(numpy.clip((imray / imray.max()).reshape(r, c, 3) + g, 0, 1) * 255)

        return Image.fromarray(i)
    return gaussian_noise_pass


def gaussian_noise_pass(x):
    r, c = x.size
    g = numpy.random.normal(0, 0.02**.5, (r, c, 3)).reshape(r, c, 3)

    imray = numpy.array(x.getdata())
    i = numpy.uint8(numpy.clip((imray / imray.max()).reshape(r, c, 3) + g, 0, 1) * 255 )

    return Image.fromarray(i )






class Gauss(object):
    def __init__(self, mean=0, var=0.01):
        self.mean = mean
        self.var = var

    def __call__(self, x):
        r, c = x.size
        g = numpy.random.normal(self.mean, self.var ** .5, (r, c, 3)).reshape(r, c, 3)

        imray = numpy.array(x.getdata())
        i = numpy.uint8(numpy.clip((imray / imray.max()).reshape(r, c, 3) + g, 0, 1) * 255)

        return Image.fromarray(i)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, var={1})'.format(self.mean, self.var)

