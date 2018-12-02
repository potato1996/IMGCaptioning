import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class Decoder(nn.Module):
    """ Decoder part(training) -- A RNN Decoder to produce the target captioning """

    def __init__(self, vocab_size, input_size=512, hidden_size=1024, num_layers=1, max_dec_len=16):
        """
        Args:
            vocab_size (int) - Size of the vocabulary => given by xxx.py
            input_size (int) - Default: 512 - Size of the input to the LSTM
            hidden_size (int) - Default: 1024 - Size of the output(and also the size of hidden state) of the LSTM
            num_layers (int) - Default: 1 - Number of layers in LSTM
            max_dec_len (int) - Default: 16 - Max decoding length
    
        Returns:
            None
        """
        super(Decoder, self).__init__()
        """ For LSTM, embedding size = input size, output size = state size = hidden size """

        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_dec_len = max_dec_len

        """1. input embedding layer convert the input word index to a vector - word2vec"""
        self.input_embedding = nn.Embedding(vocab_size, input_size)

        """2. LSTM layers, we will need to feed the output from Encoder as the first input to the LSTM, follow by <start> and other words"""
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        """3. A single FC layer at the output of LSTM, mapping back into word"""
        self.output_fc = nn.Linear(hidden_size, vocab_size)

    def init_weights(self):
        initrange = 0.1

        self.input_embedding.weight.data.uniform_(-initrange, initrange)

        self.output_fc.weight.data.uniform_(-initrange, initrange)
        self.output_fc.bias.data.fill_(0)

    def forward(self, img_embedding, input_caption, input_caption_lengths):
        """
        Args:
            img_embedding (Tensor)       - with size (batch_size, input_size)
            input_caption (Tensor)       - with size (batch_size, max_seq_len)
            input_caption_lengths (List) - Indicate the VALID lengths in the second dimension in input_caption. Thus len(input_caption_lengths) should be batch_size
    
        Returns:
            outputs (Tensor) - result of current batch of sequences, with size (batch_size, max_seq_len + 1, hidden_size)
        """

        # 0. Size checking
        batch_size = img_embedding.size(0)
        assert img_embedding.size(1) == self.input_size, "ERROR: img embedding size mismatch"
        assert input_caption.size(0) == batch_size, "ERROR: input caption batch size mismatch"
        assert len(input_caption_lengths) == batch_size, "ERROR: input_caption_lengths size mismatch"

        # 1. Embed input caption(indices) into features
        input_embedding = self.input_embedding(input_caption)  # (batch, max_len, input_size)

        # 2. put image features as the first input
        embeddings = torch.cat((img_embedding.unsqueeze(1), input_embedding), 1)  # (batch, max_len + 1, input_size)

        # (3). Wo don't need to sort them here. We have already sorted them in out data loader 
        # input_caption_lengths, perm_index = input_caption_lengths.sort(0, decending=True)
        # embeddings = embeddings[perm_index]

        # 4. Pack the sequence length into the input of LSTM
        packed = pack_padded_sequence(embeddings, input_caption_lengths, batch_first=True)

        # 5. flow through LSTM
        outputs, _ = self.lstm(packed)

        # 6. Unpack the sequence
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)  # (batch, max_len + 1, hidden_size)

        # (7). Corresponding to 3, we don't need to sort them back
        # _, unperm_index = perm_index.sort(0)
        # outputs    = outputs[unperm_index]  

        # 8. map back into vocab..
        outputs = self.output_fc(outputs)  # (batch, max_len + 1, vocab_size)

        # Maybe we will need to put softmax here?

        return outputs

    def sample(self, img_embedding, beam_width = 1):
        """ 
        Inference code for Decoder
            - due to the nature of LSTM, we need to use a complete different buch of code
        """
        # During inference we only allows batch_size = 1
        assert img_embedding.size(0) == 1, "ERROR: only allows batch_size=1 at inference time"
        
        if beam_width > 1:
            return self.sample_beam(img_embedding, beam_width)

        """ The codes below uses greedy search """

        hiddens = None
        prediction_ids = []
        inputs = img_embedding.unsqueeze(1)  # (1, 1, input_size)
        for i in range(self.max_dec_len + 1):
            """ produce the prediction of current symbol """
            if i == 0:
                outputs, hiddens = self.lstm(inputs)
            else:
                if i == 1:
                    # Assuming that 1 is the index of <start>
                    inputs = torch.ones((1, 1), dtype=torch.long, requires_grad=False).cuda()
                    inputs = self.input_embedding(inputs)  # (1, 1, input_size)

                outputs, hiddens = self.lstm(inputs, hiddens)  # (1, 1, hidden_size)

                outputs = self.output_fc(outputs.view(-1))  # (vocab_size)
                _, predicted = outputs.max(0)  # (1)
                prediction_ids.append(predicted)

                """ feed current symbol as the input of the next symbol """
                inputs = self.input_embedding(predicted.view(1, 1))  # (1, 1, input_size)
                #inputs = inputs.unsqueeze(1)  # (1, 1, input_size)

        prediction_ids = torch.stack(prediction_ids, 0)  # (max_dec_len)

        return prediction_ids
    
    def sample_beam(self, img_embedding, beam_width):
        seq = torch.LongTensor(self.max_dec_len).zero_()
        seqLogprobs = torch.FloatTensor(self.max_dec_len)

        # 1. the first input should be image
        inputs = img_embedding.expand(beam_width, self.input_size).unsequeeze(1) # (beam_width, 1, input_size)

        # 2. run the first input through the lstm
        outputs, hiddens = self.lstm(inputs)

        pass

           



class BeamSearchDecoder(Decoder):
    """ Beam Search decoder(inference) -- Use the same parameters as Decoder, but apply beam search techniques instead of gready search """

    def sample(self, img_embedding):
        """ Override the sample method with beam search, here we assume that the batch size is 1 """
