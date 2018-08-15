import chainer
from chainer import Parameter
import chainer.functions as F
import chainer.links as L
from chainer import Parameter


class Attn_Model(chainer.Chain):

    def __init__(self, attn_size):
        '''
		super(Attn_Model, self).__init__(

			self.l1 = L.linear(None, hidden_size)
		)

		#self.u = chainer.initializers.Normal(scale=0.5)
		'''
        super(Attn_Model, self).__init__()

        with self.init_scope():
            # self.inputs = inputs
            self.attn_size = attn_size
            self.word_attention = Attn_module(self.attn_size)
            self.sentence_attention = Attn_module(self.attn_size)

    def __call__(self, inputs):
        batch_size, max_doc_len, max_seq_len, embedding_size = inputs.shape
        new_inputs = F.reshape(inputs, (batch_size * max_doc_len, max_seq_len, embedding_size))
        # print("shape:{}".format(new_inputs.shape))
        word_attention_output = self.word_attention(new_inputs)
        # print(word_attention_output.data.shape)
        sentence_attention_inputs = F.reshape(word_attention_output, (batch_size, max_doc_len, embedding_size))

        # print("sentence_attention_inputs.shape:{}".format(sentence_attention_inputs.shape))
        sentence_attention_output = self.sentence_attention(sentence_attention_inputs)
        # print("sentence_attention_output.shape:{}".format(sentence_attention_output.shape))

        return F.reshape(sentence_attention_output, (batch_size, embedding_size))
    # word_mask = []


class Attn_module(chainer.Chain):

    def __init__(self, attn_size):
        super(Attn_module, self).__init__()

        with self.init_scope():
            self.attn_size = attn_size
            # "self.ux = Variable(self.xp.random.normal(0, 0.5, (self.attn_size, 1), dtype=float))
            self.ux = Parameter(
                initializer=self.xp.random.normal(0, 0.5, (self.attn_size, 1)).astype('f')
            )
            self.l = L.Linear(None, self.attn_size)

    def __call__(self, inputs):
        # input shape:b*max_doc,max_seq,emb_size

        # assert(inputs.shape[-1] == mask.shape[-1])
        # assert(inputs.shape[0] == mask.shape[0])
        # print("inputs.shape:".format(inputs.shape))
        b_d_size, seq_len, emb_size = inputs.shape

        new_input = self.l(F.reshape(inputs, (b_d_size * seq_len, emb_size)))

        # print(new_input.shape)
        # print(F.tanh(new_input).shape)
        # print(F.matmul(F.tanh(new_input), self.ux).shape)

        weights = F.softmax(
            F.reshape(
                F.matmul(
                    F.tanh(
                        F.reshape(
                            new_input,
                            (b_d_size, seq_len, self.attn_size)
                        )
                    ),
                    F.broadcast_to(
                        self.ux,
                        (b_d_size, self.attn_size, 1)
                    )
                ),
                (b_d_size, seq_len)
            )
        )  # b*max_doc, max_seq

        return F.matmul(F.expand_dims(weights, axis=2), inputs, transa=True)
