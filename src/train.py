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
from chainer.datasets import tuple_dataset
from chainer.dataset.convert import concat_examples
from chainer.cuda import to_cpu

from chainer.functions.loss.mean_squared_error import mean_squared_error

from sklearn.feature_extraction.text import TfidfVectorizer

from hyperopt import fmin, tpe, hp
from hyperopt.pyll import scope

from argparse import ArgumentParser
from utils import *
from embedding import *
from mlp import *
import ast

def main(params):
#def main():

	parser = ArgumentParser()
	parser.add_argument('--gpu', '-g', default=-1, type=int,
						help='GPU ID')
	parser.add_argument('--embedding_file', '-e', metavar="FILE", default=None)
	parser.add_argument('--training_file', metavar="FILE", default=None)
	parser.add_argument('--test_file', metavar="FILE", default=None)
	parser.add_argument('--voc_limit', metavar="INT", type=int, default=100000)
	parser.add_argument('--num_epoch', metavar="INT", type=int, default=10)
	parser.add_argument('--first_layer', '-f', metavar="INT", type=int, default=100)
	parser.add_argument('--second_layer', '-s', metavar="INT", type=int, default=20)
	parser.add_argument('--cut_max_length', '-c', metavar="INT", type=int, default=None)
	
	######
	# for fine tuning by hyperopt
	f_layer = int(params['f_layer_num'])
	s_layer = int(params['s_layer_num'])
	cut_length = int(params['cut_length'])
	print("f_layer={0}".format(f_layer))
	print("s_layer={0}".format(s_layer))
	######
	
	args = parser.parse_args()

	if args.embedding_file is None:
		args.embedding_file = "."

	print(args.gpu)
	cuda.check_cuda_available()
	
	if args.embedding_file is not None:
		print(args.embedding_file)
	if args.training_file is not None:
		print(args.training_file)
	if args.test_file is not None:
		print(args.test_file)
	
	print(args.voc_limit)
	print(args.num_epoch)

	train_data, train_label = Utils.load_data(args.training_file)
	test_data, test_label = Utils.load_data(args.test_file)

#############################
	
	#Approach 1: setting word embedding layer 
	embedding_loader = Embedding_Loader(embedding_file_path = args.embedding_file)
	embedding_l = embedding_loader.load_embedding(voc_limit=30000)

	train_max_length, train_data = embedding_loader.seq_to_ids(train_data)
	test_max_length, test_data = embedding_loader.seq_to_ids(test_data)

	max_length = train_max_length if train_max_length > test_max_length else test_max_length

	train_data = Utils.zero_padding(train_data, max_length)
	test_data = Utils.zero_padding(test_data, max_length)

	#if args.cut_max_length is not None:
		#train_data = Utils.cut_seq(train_data, args.cut_max_length)
		#test_data = Utils.cut_seq(test_data, args.cut_max_length)
	if cut_length != 0:
		train_data = Utils.cut_seq(train_data, cut_length)
		test_data = Utils.cut_seq(test_data, cut_length)
		
	
	train_data = embedding_l(train_data)
	test_data = embedding_l(test_data)

	
	train_data = F.reshape(train_data, [-1, train_data.shape[2] * train_data.shape[1]])
	#train_data = tuple_dataset.TupleDataset(train_data.array, train_label.reshape(-1,1))
	#train_label = np.eye(4)[train_label].astype(np.int32)
	#print(train_label.shape)
	#print(train_data.shape)
	train_data = tuple_dataset.TupleDataset(train_data.array, train_label)
	#train_data = tuple_dataset.TupleDataset(train_data.array, train_label)
	test_data = F.reshape(test_data, [-1, test_data.shape[2] * test_data.shape[1]])
	#test_data = tuple_dataset.TupleDataset(test_data.array, test_label.reshape(-1,1))
	#test_label = np.eye(4)[test_label].astype(np.int32)
	#print(test_label.shape)
	#print(test_data.shape)
	test_data = tuple_dataset.TupleDataset(test_data.array, test_label)
	#test_data = tuple_dataset.TupleDataset(test_data.array, test_label)
	
#############################
	'''
	#Approach 2: extract tfidf feature
	vectorizer = TfidfVectorizer()
	vectorizer.fit(train_data)

	train_data_vectorized = np.array(vectorizer.transform(train_data).toarray(), dtype=np.float32)
	test_data_vectorized = np.array(vectorizer.transform(test_data).toarray(), dtype=np.float32)

	train_data = tuple_dataset.TupleDataset(train_data_vectorized, train_label)
	test_data = tuple_dataset.TupleDataset(test_data_vectorized, test_label)
	
	#print(train_data_vectorized.shape)
	#print(train_data_vectorized.dtype.kind)
	#print(test_data_vectorized.shape)
	'''
##########################
	batch_size = 32
	#batch_size = 128
	train_iter = chainer.iterators.SerialIterator(train_data, batch_size)
	test_iter = chainer.iterators.SerialIterator(test_data, batch_size, repeat=False, shuffle=False)

	#model = L.Classifier(MLP(100, 50, 1))
	#model = L.Classifier(MLP(5, 2, 1), lossfun=mean_squared_error)
	#model = L.Classifier(MLP(50, 20, 4))
	model = L.Classifier(MLP(f_layer, s_layer, 4))
	#model = L.Classifier(MLP(args.first_layer, args.second_layer, 4))
	#model = MLP(100, 50, 1)
	#model.to_gpu(args.gpu)
	
	#optimizer = chainer.optimizers.SGD()
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)

	#print("optimizer")
	
	updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
	trainer = training.Trainer(updater, (args.num_epoch, 'epoch'), out="result")

	trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
	#trainer.extend(extensions.dump_graph('main/loss'))
	#trainer.extend(extensions.snapshot(), trigger=(args.num_epoch, 'epoch'))
	trainer.extend(extensions.LogReport())
	trainer.extend(extensions.PrintReport(
		['epoch', 'main/loss', 'validation/main/loss',
		'main/accuracy', 'validation/main/accuracy']))
	'''
	trainer.extend(extensions.PlotReport(
		['main/loss', 'validation/main/loss'],
		'epoch', file_name='loss.png'))
	trainer.extend(extensions.PlotReport(
		['main/accuracy', 'validation/main/accuracy'],
		'epoch', file_name='accuracy.png'))
	trainer.extend(extensions.ProgressBar())
	'''
	print("start running...")
	trainer.run()
	print("finished running...")	

######
	
	# for fine tuning by hyperopt
	'''
	valid_data = trainer._extensions['PlotReport'].extension._data
	loss_data = [data for i, data in valid_data['validation/main/loss']]
	best10_loss = sorted(loss_data)[:10]
	return sum(best10_loss) / 10
	'''
	with open('result/log', 'r') as f:
		tmp_result_str = f.read()
	tmp_result_list = ast.literal_eval(tmp_result_str)
        # print(tmp_result_list)
        # print(type(tmp_result_list))
	loss_data = []
	for (i, tmp_dict) in enumerate(tmp_result_list):
		loss_data.append(tmp_dict['validation/main/loss'])

	if len(loss_data) > 9:
		best_loss = sorted(loss_data)[:10]
		return sum(best_loss) / 10
	else:
		best_loss = sorted(loss_data)[:]
		return sum(best_loss) / len(loss_data)
	
######
	'''
	while train_iter.epoch < args.num_epoch:
		
		print("train_iter.epoch=%d" % train_iter.epoch)
		# ---------- 学習の1 epoch ----------
		train_batch = train_iter.next()
		# minibatch の事例列を連結して配列にします
		print("train_batch")
		print(train_batch)
		#x, t = concat_examples(train_batch, device=args.gpu)
		x, t = concat_examples(train_batch, device=args.gpu)
		print("xxx")
		print(x)
		print("concat_examples")
		#x = Variable(x) # (128, 784)
		#print("x.data.dtype.kind = {0}".format(x.data.dtype.kind))
		t = Variable(t) # (128,)
		
		print("2")
		# 損失の計算
		loss = model(x, t) # (128, 10), (128,)

		print("3")
		# 勾配の計算
		model.cleargrads()
		loss.backward()

		print("4")
		# パラメータの更新
		optimizer.update()
		# --------------- ここまで ----------------

		# 1エポック終了ごとにValidationデータに対する予測精度を測って、
		# モデルの汎化性能が向上していることをチェックしよう
		if train_iter.is_new_epoch:  # 1 epochが終わったら

			# 損失の表示
			print('epoch:{:02d} train_loss:{:.04f} '.format(
			train_iter.epoch, float(to_cpu(loss.data))), end='')

			test_losses = []
			test_accuracies = []
			while True:
				test_batch = test_iter.next()
				x_test, t_test = concat_examples(test_batch, device=args.gpu)

				# テストデータをforward
				y_test = model.predict(x_test)

				# 損失を計算
				loss_test = F.softmax_cross_entropy(y_test, t_test)
				test_losses.append(to_cpu(loss_test.data))

				# 精度を計算
				accuracy = F.accuracy(y_test, t_test)
				accuracy.to_cpu()
				test_accuracies.append(accuracy.data)
				
				if test_iter.is_new_epoch:
					test_iter.epoch = 0
					test_iter.current_position = 0
					test_iter.is_new_epoch = False
					test_iter._pushed_position = None
					break

			print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
				np.mean(test_losses), np.mean(test_accuracies)))
	'''

if __name__ == '__main__':
	#main(f_layer_num=200, s_layer_num=100)
	#main()
	
	###
	# for fine tuning by hyperopt
	space = {'f_layer_num': scope.int(hp.quniform('f_layer_num', 100, 300, 50)),
			 's_layer_num': scope.int(hp.quniform('s_layer_num', 10, 200, 10)),
			 'cut_length': scope.int(hp.quniform('cut_length', 100, 1500, 50))
			}
	best = fmin(main, space, algo=tpe.suggest, max_evals=200)
	print("best parameters", best)
	
	###

