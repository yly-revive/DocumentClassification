import os
import codecs
import numpy as np

import chainer.links as L
from utils import *


class Embedding_Loader(object):

    def __init__(self, **kwargs):

        super(Embedding_Loader, self).__init__()

        self.dims = 0
        self.key2id = dict()
        # self.embedding = None
        self.embedding_file_path = kwargs.pop('embedding_file_path', '.')

    # @classmethod
    def load_embedding(self, voc_limit=100000):

        if os.path.isfile(self.embedding_file_path) == False:
            return

        # with open(self.embedding_file_path, 'r')
        print(self.embedding_file_path)
        embedding_file = open(self.embedding_file_path, 'rb')
        first_line = embedding_file.readline().decode('ascii')
        print(first_line)
        voc_size, dims = first_line.rstrip().split(" ", 1)

        voc_size = int(voc_size)

        if 0 < voc_limit < voc_size:
            voc_size = voc_limit

        self.dims = int(dims)
        embedding = L.EmbedID(voc_size + 2, self.dims, ignore_label=-1)

        one_dim_bin_len = np.dtype(np.float32).itemsize * self.dims

        '''
        #for unk words
        unk_id = 0
        self.key2id[Utils.UNK_CONSTANT] = unk_id
        embedding.W.data[unk_id] = np.random.uniform(-0.25, 0.25, self.dims)
        '''
        eos_id = 0
        self.key2id[Utils.EOS_CONSTANT] = eos_id
        embedding.W.data[eos_id] = np.random.uniform(-0.25, 0.25, self.dims)

        for line_no in range(voc_size):

            word_arr = []

            while True:

                ch = embedding_file.read(1)
                if ch == b' ':
                    break
                if ch != b'\n':
                    word_arr.append(ch)

            # word = unicode(b''.join(word_arr), encoding="utf8", errors="ignore")
            word = b''.join(word_arr)
            word = word.decode('utf-8')
            word_id = len(self.key2id)
            self.key2id[word] = word_id
            embedding.W.data[word_id] = np.fromstring(embedding_file.read(one_dim_bin_len), dtype=np.float32)

        # for unk words
        unk_id = len(self.key2id)
        self.key2id[Utils.UNK_CONSTANT] = unk_id
        embedding.W.data[unk_id] = np.random.uniform(-0.25, 0.25, self.dims).astype('f')

        return embedding

    def seq_to_ids(self, seqs):

        assert (self.key2id != None)

        id_seqs = []
        max_length = 0
        for line in seqs:
            words_id = [self.key2id[word] if word in self.key2id else self.key2id[Utils.UNK_CONSTANT] for word in
                        line.strip().split()]

            # id_seqs.append(' '.join(words_id))
            # max_length = len(words_id) if len(words_id) > max_length
            if len(words_id) > max_length:
                max_length = len(words_id)
            id_seqs.append(words_id)

        # print(len(words_id))
        # print("seq_to_ids:")
        # print(len(id_seqs))
        # print(max_length)

        # id_seqs = Utils.zero_padding(id_seqs, max_length)
        # print(id_seqs)

        return max_length, id_seqs

    def documents_to_ids(self, docs):

        max_seq_length = 0
        max_doc_length = 0

        id_docs = []
        # print(len(doc))
        for seqs in docs:
            m_len, id_seqs = self.seq_to_ids(seqs)
            if max_seq_length < m_len:
                max_seq_length = m_len
            # print(len(id_seqs))
            id_docs.append(id_seqs)
            # print(np.array(id_docs)[0].shape)
            if len(id_seqs) > max_doc_length:
                max_doc_length = len(id_seqs)

        return max_seq_length, max_doc_length, id_docs


# def embedding_words(self, id_seqs):


'''
if __name__ == '__main__':
	embedding = Embedding_Loader()	
'''
