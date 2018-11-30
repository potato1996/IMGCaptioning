import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class Decoder(nn.Module):
    """ Decoder part(training) -- A RNN Decoder to produce the target captioning """

    def __init__(self, feature_size, input_size, hidden_size, vocab_size, num_layers, max_dec_len):
        """
        Args:
            feature_size (int) - img feature size
            input_size (int) - Size of the input to the LSTM
            hidden_size (int) - Size of the output(and also the size of hidden state) of the LSTM
            vocab_size (int) - Size of the vocabulary => given by xxx.py
            num_layers (int) - Number of layers in LSTM
            max_dec_len (int) - Max decoding length
    
        Returns:
            None
        """    
        super(Decoder, self).__init__()
        """ For LSTM, embedding size = input size, output size = state size = hidden size """

        self.feature_size = feature_size
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.vocab_size   = vocab_size
        self.num_layer    = num_layers
        self.max_dec_len  = max_dec_len

        """1. mapping image feature into input_size so that it would be the input of first sequence"""
        self.img_embedding = nn.Linear(feature_size, input_size)

        """2. input embedding layer convert the input word index to a vector - word2vec"""
        self.input_embedding = nn.Embedding(vocab_size, input_size)

        """3. LSTM layers, we will need to feed the output from Encoder as the initial hidden state of Decoder"""
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        """4. A single FC layer at the output of LSTM, mapping back into word"""
        self.output_fc = nn.Linear(hidden_size, vocab_size)

    def init_weights(self):
        initrange = 0.1

        self.img_embedding.weight.data.normal_(0.0, 0.02)
        self.img_embedding.bias.data.fill_(0)
        
        self.input_embedding.weight.data.uniform_(-initrange, initrange)

        self.output_fc.weight.data.uniform_(-initrange, initrange)
        self.output_fc.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = self.lstm.data
        return(Variable(weight.new(self.num_layer, batch_size, self.hidden_size)._zero()),
               Variable(weight.new(self.num_layer, batch_size, self.hidden_size)._zero()))


    def forward(self, features, input_caption, input_caption_lengths):
        batch_size = features.size(0)
        
        input_embedding = self.input_embedding(input_caption) # (batch, max_len, input_size)
        img_embedding   = self.img_embedding(features)        # (batch, input_size)
        
        embeddings = torch.cat((img_embedding.unsqueeze(1), input_embedding), 1) # (batch, max_len + 1, input_size)
        
        # We need to sort the inputs before pack them
        input_caption_lengths, perm_index = input_caption_lengths.sort(0, decending=True)
        embeddings = embeddings[perm_index]

        packed     = pack_padded_sequence(embeddings, input_caption_lengths, batch_first=True)
        
        hiddens    = self.init_hidden(batch_size)
        outputs, _ = self.lstm(packed, hiddens)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # Order them back..
        _, unperm_index = perm_index.sort(0)
        outputs    = outputs[unperm_index]  

        outputs    = self.output_fc(outputs[0])

        return outputs

    def sample(self, features, lengths, hiddens=None):
        """ The codes below uses greedy search """
        sampled_ids     = []
        img_embedding   = self.img_embedding(features)        # (1, input_size)
        inputs          = img_embedding.unsqueeze(1)          # (1, 1, input_size)
        for _ in range(self.max_dec_len):
            """ produce the prediction of current symbol """
            outputs, hiddens = self.lstm(inputs, hiddens)
            outputs          = self.output_fc(outputs)
            _, predicted     = outputs.max(1)
            sampled_ids.append(predicted)
            
            """ feed current symbol as the input of the next symbol """
            inputs           = self.input_embedding(predicted)
            inputs           = inputs.unsqueeze(1)

        sampled_ids     = torch.stack(sampled_ids, 1)
        return sampled_ids

class BeamSearchDecoder(Decoder):
    """ Beam Search decoder(inference) -- Use the same parameters as Decoder, but apply beam search techniques instead of gready search """
    def sample(self, features, lengths):
        """ Override the sample method with beam search, here we assume that the batch size is 1 """