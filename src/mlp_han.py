import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F

import chainer.links as L
from chainer import reporter
from chainer.training import extensions


class MLP_Han(chainer.Chain):

    def __init__(self, output_layer_units=1, use_bn=False):
        super(MLP_Han, self).__init__()

        with self.init_scope():
            self.use_bn = use_bn

            self.l1 = L.Linear(None, output_layer_units)
            '''
            self.l3 = L.Linear(None, second_layer_units * 2)
            self.l4 = L.Linear(None, second_layer_units)
            self.l5 = L.Linear(None, output_layer_units)
            '''

    # def __call__(self, data, label):
    def __call__(self, data):
        # print("in __call__")

        pred = self.predict(data)
        '''
        # use L.Classifier instead
        loss = F.softmax_cross_entropy(pred, label)

        return loss
        '''
        # print("in __call__")
        # return pred
        # print(pred.data[0].dtype)
        return pred

    def predict(self, data):
        return self.l1(data.data.astype(self.xp.float32))
        '''
        h3 = F.relu(self.l3(h2))

        h4 = F.relu(self.l4(h3))

        #return self.l3(h2)
        return self.l5(h4)
        '''

# return self.l3(h2)
