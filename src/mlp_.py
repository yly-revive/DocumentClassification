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

class MLP_(chainer.Chain):

	def __init__(self, first_layer_units, second_layer_units, output_layer_units = 1):

		super(MLP_, self).__init__()

		with self.init_scope():

			self.l1 = L.Linear(None, first_layer_units)
			self.l2 = L.Linear(None, second_layer_units)
			self.l3 = L.Linear(None, output_layer_units)

	def __call__(self, data, label):

		pred = self.predict(data)

		'''
		# use L.Classifier instead
		loss = F.softmax_cross_entropy(y, t)

		return loss
		'''
		print("in __call__")
		return pred


	def predict(self, data):

		h1 = F.relu(self.l1(data))

		h2 = F.relu(self.l2(h1))

		return self.l3(h2)









