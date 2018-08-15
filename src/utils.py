import os
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

np.random.seed(1234)

class Utils(object):
    UNK_CONSTANT = 'UNK'
    EOS_CONSTANT = 'EOS'
    PAD_CONSTANT = 'PAD'

    def __init__(self):
        super(Utils, self).__init__()

    @classmethod
    def load_data(cls, data_path):

        if os.path.isfile(data_path) is False:
            return None

        data, label = [], []
        with open(data_path, 'r') as data_file:

            for line in data_file:
                tmp_label, tmp_data = line.strip().split(',')
                data.append(tmp_data)
                label.append(tmp_label)

        # return np.array(data), np.array(label, 'i')
        return np.array(data), np.array(label, 'i')

    @classmethod
    def load_data_seq(cls, data_path):

        if os.path.isfile(data_path) is False:
            return None

        data, label = [], []
        with open(data_path, 'r') as data_file:

            for line in data_file:
                tmp_label, tmp_data = line.strip().split(',')
                # data.append(tmp_data)
                label.append(tmp_label)

                tmp_data_seqs = tmp_data.split('。')
                tmp_seq = [seq for seq in tmp_data_seqs]
                data.append(tmp_seq)

            np_data = np.array(data)
            np_label = np.array(label, 'i')

            size = len(np_data)
            indics = np.arange(size)
            np.random.shuffle(indics)

            np_data = np_data[indics]
            np_label = np_label[indics]

        #return np.array(data), np.array(label, 'i')
        return np_data, np_label

    '''
    @classmethod
    def seq_to_ids(seqs, key2id_map):

        assert(key2id_map != None)

        id_seqs = []
        for line in seqs:
            words_id = [[key2id_map[word] if word in key2id_map else key2id_map[UNK_CONSTANT]]]
            
            id_seqs.append(' '.join(words_id))

        return seq_to_ids
    '''

    @classmethod
    def print_scores(sefl, y_test, y_test_pred):

        f_score = f1_score(y_test, y_test_pred, average='macro')
        accuracy = accuracy_score(y_test, y_test_pred)
        print('F-score : %f' % f_score)
        print('Accuracy: %f' % accuracy)

    @classmethod
    def zero_padding(self, seqs, max_length):
        padded_line = []
        for line in seqs:
            line = line + [-1] * max(max_length - len(line), 0)
            padded_line.append(line)

        # return np.array(padded_line)
        # print(len(padded_line))
        # print(len(padded_line[0]))
        return np.array(padded_line, 'i')

    '''
    @classmethod
    def zero_padding_doc(self, doc, max_seq_length, max_doc_length):

        padded_doc = []
        line_stub = [0] * max_seq_length
        doc_lvl_len = []
        #seq_lvl_len = []

        for seqs in doc:
            padded_line = []
            #doc_len = []
            seq_lvl_len = []
            for line in seqs:
                # len of sentence
                seq_lvl_len.append(len(line))
                # pad each sentence
                line = line + [0] * max(max_seq_length - len(line), 0) 
                padded_line.append(line)
            doc_lvl_len.append(seq_lvl_len)

            add_line = [line_stub] *  max(max_doc_length - len(padded_line), 0)

            padded_line = padded_line + add_line
            padded_doc.append(padded_line)

        #return np.array(padded_line)
        #print(len(padded_line))
        #print(len(padded_line[0]))
        return np.array(padded_doc, 'f'), doc_lvl_len
    '''

    @classmethod
    def zero_padding_doc(self, doc, max_seq_length, max_doc_length):

        padded_doc = []
        line_stub = [-1] * max_seq_length

        for seqs in doc:
            padded_line = []

            for line in seqs:
                # pad each sentence
                line = line + [-1] * max(max_seq_length - len(line), 0)
                padded_line.append(line)

            add_line = [line_stub] * max(max_doc_length - len(padded_line), 0)

            padded_line = padded_line + add_line
            padded_doc.append(padded_line)

        '''
        print(len(padded_doc))
        print(len(padded_doc[0]))
        print(len(padded_doc[0][0]))

        with open('test.txt', 'w') as f:
            for i in range(len(padded_doc)):
                for j in range(len(padded_doc[i])):
                    f.write('i:{}, j:{}, k:{}\n'.format(i,j,len(padded_doc[i][j])))
                f.write('\n')
        '''

        res = np.array(padded_doc, 'i')
        # return np.array(padded_doc, 'f')

        return res

    @classmethod
    def zero_padding_doc_cut(self, doc, max_seq_length, max_doc_length, cut_seq_len=500, cut_line_len=100):

        padded_doc = []

        t_max_seq_length = cut_seq_len if cut_seq_len < max_seq_length else max_seq_length
        t_max_line_length = cut_line_len if cut_line_len < max_doc_length else max_doc_length

        line_stub = [-1] * t_max_seq_length

        for seqs in doc:
            padded_line = []

            for line in seqs:
                # pad each sentence
                line = line + [-1] * max(t_max_seq_length - len(line), 0)
                line = line[:t_max_seq_length]
                padded_line.append(line)

            add_line = [line_stub] * max(t_max_line_length - len(padded_line), 0)

            padded_line = padded_line + add_line
            padded_line = padded_line[:t_max_line_length]
            padded_doc.append(padded_line)

        '''
        print(len(padded_doc))
        print(len(padded_doc[0]))
        print(len(padded_doc[0][0]))

        with open('test.txt', 'w') as f:
            for i in range(len(padded_doc)):
                for j in range(len(padded_doc[i])):
                    f.write('i:{}, j:{}, k:{}\n'.format(i,j,len(padded_doc[i][j])))
                f.write('\n')
        '''

        res = np.array(padded_doc, 'i')

        return res

    @classmethod
    def cut_seq(self, seqs, max_length):
        cut_line = []
        for line in seqs:
            line = line[:max_length]
            cut_line.append(line)

        return np.array(cut_line, 'i')
