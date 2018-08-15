from Attn_model import *
from mlp import *
#from mlp_han import *


class Han(chainer.Chain):

    def __init__(self, doc_len, seq_len, emb_size):

        super(Han, self).__init__()

        with self.init_scope():
            print("in Han")
            self.doc_len = doc_len
            self.seq_len = seq_len
            self.emb_size = emb_size
            self.bilstm = L.NStepBiLSTM(
                1,
                emb_size,
                128,
                1
            )
            self.attn_model = Attn_Model(50)
            #self.mlp = MLP_Han(4)
            #self.mlp = MLP(200, 100, 4, True)
            #self.mlp = MLP(200, 100, 50, 4, True)
            self.mlp = MLP(512, 128, 64, 4, True)

    def __call__(self, inputs):
        # print("type : {}",format(type(inputs)))
        batch_size = inputs.shape[0]
        lstm_input = []

        inputs = F.reshape(inputs, (inputs.shape[0], self.doc_len, self.seq_len, self.emb_size))

        '''
        for i in range(batch_size):
            for j in range(self.doc_len):
                lstm_input.append(inputs[i, j])

        _, _, lstm_output = self.bilstm(None, None, lstm_input)


        attention_inputs = F.stack(lstm_output, axis=0)
        
        attention_inputs = F.reshape(attention_inputs, (batch_size, self.doc_len, self.seq_len, self.emb_size))
        
        attention_output = self.attn_model(attention_inputs)
        '''
        attention_output = self.attn_model(inputs)
        #attention_output = F.average(F.average(inputs, axis=3), axis=2)

        pred = self.mlp(attention_output)

        return pred
