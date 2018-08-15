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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from config import *

from argparse import ArgumentParser
from utils import *
from embedding import *
from mlp import *
from HAN import *
import  datetime

def main():
    parser = ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID')
    parser.add_argument('--embedding_file', '-e', metavar="FILE", default=None)
    parser.add_argument('--training_file', metavar="FILE", default=None)
    parser.add_argument('--test_file', metavar="FILE", default=None)
    parser.add_argument('--voc_limit', metavar="INT", type=int, default=100000)
    parser.add_argument('--num_epoch', metavar="INT", type=int, default=10)

    args = parser.parse_args()
    '''
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
	'''

    # train_data, train_label = Utils.load_data_seq(args.training_file)
    # test_data, test_label = Utils.load_data_seq(args.test_file)
    train_data, train_label = Utils.load_data_seq(TRAINING_FILE)
    test_data, test_label = Utils.load_data_seq(TEST_FILE)

    # embedding_loader = Embedding_Loader(embedding_file_path = args.embedding_file)
    embedding_loader = Embedding_Loader(embedding_file_path=EMBEDDING_FILE)
    embedding_l = embedding_loader.load_embedding(voc_limit=VOC_LIMIT)
    embedding_l.disable_update()

    print("1")

    train_mask = []
    test_mask = []

    train_max_seq_length, train_max_doc_length, train_data = embedding_loader.documents_to_ids(train_data)
    test_max_seq_length, test_max_doc_length, test_data = embedding_loader.documents_to_ids(test_data)

    print("2")
    # print(train_max_length)
    # print(test_max_length)
    # print(np.array(train_data).shape)
    # print(np.array(test_data).shape)
    max_seq_length = train_max_seq_length if train_max_seq_length > test_max_seq_length else test_max_seq_length
    max_doc_length = train_max_doc_length if train_max_doc_length > test_max_doc_length else test_max_doc_length

    print('max_doc_length:{}'.format(max_doc_length))
    print('max_seq_length:{}'.format(max_seq_length))

    print(len(train_data))

    print(len(test_data))

    train_data = Utils.zero_padding_doc_cut(train_data, max_seq_length, max_doc_length, 100, 40)
    test_data = Utils.zero_padding_doc_cut(test_data, max_seq_length, max_doc_length, 100, 40)

    print("3")
    print(train_data.shape)
    print(test_data.shape)

    '''
	train_max_length, train_data = embedding_loader.seq_to_ids(train_data)
	test_max_length, test_data = embedding_loader.seq_to_ids(test_data)

	max_length = train_max_length if train_max_length > test_max_length else test_max_length

	train_data = Utils.zero_padding(train_data, max_length)
	test_data = Utils.zero_padding(test_data, max_length)
	'''

    train_data = embedding_l(train_data)
    test_data = embedding_l(test_data)

    train_size = train_data.shape[0]
    validation_size = int(train_size * VALIDATION_RATIO)

    validation_data = train_data[:validation_size]
    train_data = train_data[validation_size:]
    validation_label = train_label[:validation_size]
    train_label = train_label[validation_size:]

    print("end embedding_l")

    print(train_data.shape)
    print(validation_data.shape)
    print(test_data.shape)

    _, doc_len, seq_len, emb_len = train_data.shape

    train_data = F.reshape(train_data, [-1, train_data.shape[2] * train_data.shape[1] * train_data.shape[3]])
    # train_data = tuple_dataset.TupleDataset(train_data.array[:,:1000], train_label)
    train_data = tuple_dataset.TupleDataset(train_data.array, train_label)

    validation_data = F.reshape(validation_data,
                                [-1, validation_data.shape[2] * validation_data.shape[1] * validation_data.shape[3]])
    # train_data = tuple_dataset.TupleDataset(train_data.array[:,:1000], train_label)
    validation_data = tuple_dataset.TupleDataset(validation_data.array, validation_label)

    test_data = F.reshape(test_data, [-1, test_data.shape[2] * test_data.shape[1] * test_data.shape[3]])
    '''
    # test_data = tuple_dataset.TupleDataset(test_data.array[:,:1000], test_label)
    test_data = tuple_dataset.TupleDataset(test_data.array, test_label)
    '''

    print("end tuple_dataset")

    batch_size = 128
    train_iter = chainer.iterators.SerialIterator(train_data, batch_size)
    # test_iter = chainer.iterators.SerialIterator(test_data, batch_size, repeat=False, shuffle=False)
    validation_iter = chainer.iterators.SerialIterator(validation_data, batch_size, repeat=False, shuffle=False)

    # model = L.Classifier(MLP(100, 50, 1))
    model = L.Classifier(Han(doc_len, seq_len, emb_len))
    # model = MLP(100, 50, 1)
    # model.to_gpu(args.gpu)

    # optimizer = chainer.optimizers.SGD()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # print("optimizer")

    # updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    # trainer = training.Trainer(updater, (args.num_epoch, 'epoch'), out="result")

    # trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    updater = training.StandardUpdater(train_iter, optimizer, device=GPU)
    trainer = training.Trainer(updater, (NUM_EPOCH, 'epoch'), out="result")

    # trainer.extend(extensions.Evaluator(test_iter, model, device=GPU))
    trainer.extend(extensions.Evaluator(validation_iter, model, device=GPU))

    # trainer.extend(extensions.dump_graph('main/loss'))
    # trainer.extend(extensions.snapshot(), trigger=(args.num_epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.extend(
        extensions.snapshot_object(model, 'best_model'),
        trigger=chainer.training.triggers.MinValueTrigger('validation/main/loss')
    )
    print("start running...")
    trainer.run()
    print("finished running...")

    print("start testing...")

    chainer.serializers.load_npz('result/best_model', model)
    model.to_cpu()
    result = model.predictor(test_data)

    pred = F.argmax(result, axis=1)
    print(f1_score(test_label.flatten(), pred.data.flatten(), average='macro'))

    FILE_NAME = 'tmp_output/tmp_output_'
    current_datetime = datetime.datetime.now()
    current_time_str = current_datetime.strftime('%Y_%m_%d_%H:%M:%S')
    filename = FILE_NAME + current_time_str + '.txt'

    with open(filename, 'w') as file:
        file.write(str(f1_score(test_label.flatten(), pred.data.flatten(), average='macro')))
        file.write('\n')
    print("finished testing...")


if __name__ == '__main__':
    main()
