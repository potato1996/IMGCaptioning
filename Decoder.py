import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class Decoder(nn.Module):
    """ Decoder part(training) -- A RNN Decoder to produce the target captioning """

    def __init__(self, vocab_size, input_size=512, hidden_size=512, num_layers=1, max_dec_len=16, drop_rate=0.2):
        """
        Args:
            vocab_size (int) - Size of the vocabulary => given by xxx.py
            input_size (int) - Default: 512 - Size of the input to the LSTM
            hidden_size (int) - Default: 512 - Size of the output(and also the size of hidden state) of the LSTM
            num_layers (int) - Default: 1 - Number of layers in LSTM
            max_dec_len (int) - Default: 16 - Max decoding length
            drop_rate (float) - Default: 0.2 - drop out rate
        
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
        self.drop_rate = drop_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        """1. input embedding layer convert the input word index to a vector - word2vec"""
        self.input_embedding = nn.Embedding(vocab_size, input_size)

        """2. LSTM layers, we will need to feed the output from Encoder as the first input to the LSTM, follow by <start> and other words"""
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        """3. A single FC layer at the output of LSTM, mapping back into word"""
        self.output_fc = nn.Linear(hidden_size, vocab_size)

        """4. Drop out layer before the laster FC"""
        self.dropout = nn.Dropout(self.drop_rate)

        self.init_weights()

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
        outputs = self.output_fc(self.dropout(outputs))  # (batch, max_len + 1, vocab_size)

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
                    inputs = torch.tensor([1], dtype=torch.long).to(self.device)
                    inputs = inputs.unsqueeze(1)  # (1, 1)
                    inputs = self.input_embedding(inputs)  # (1, 1, input_size)

                outputs, hiddens = self.lstm(inputs, hiddens)  # (1, 1, hidden_size)

                outputs = self.output_fc(outputs.squeeze(1))  # (1, vocab_size)
                _, predicted = outputs.max(1)  # (1)
                prediction_ids.append(predicted.cpu().data.tolist()[0])

                """ feed current symbol as the input of the next symbol """
                inputs = self.input_embedding(predicted.view(1, 1))  # (1, 1, input_size)
                #inputs = inputs.unsqueeze(1)  # (1, 1, input_size)

        return prediction_ids
    
    def sample_beam(self, img_embedding, beam_width):
        hypos = []
        
        # 1. the first input should be image
        inputs = img_embedding.unsqueeze(1)  # (1, 1, input_size)

        # 2. run the first input through the lstm
        _, hiddens = self.lstm(inputs)

        # 3. The first input of sentense is <start>(1)
        inputs = torch.tensor([1], dtype=torch.long).to(self.device)
        inputs = inputs.unsqueeze(1) # (1, 1)
        inputs = self.input_embedding(inputs) # (1, 1, input_size)

        # 4. run the lstm through start
        outputs, hiddens = self.lstm(inputs, hiddens) # (1, 1, hidden_size)
        outputs = self.output_fc(outputs.squeeze(1)) #(1, vocab_size)

        # Get the starters right after <start>
        track_table = [[] for i in range(beam_width)]

        outputs = outputs.view(-1)
        scores  = -F.log_softmax(outputs)
        curr_beam_scores, topk_idx = scores.topk(beam_width, largest=False) #[beam_width]

        next_word_idx = topk_idx % self.vocab_size
        prev_beam_id  = topk_idx / self.vocab_size

        for beam_id in range(beam_width):
            track_table[beam_id].append((next_word_idx[beam_id].cpu().data.tolist(), prev_beam_id[beam_id].cpu().data.tolist()))
        hiddens = (hiddens[0].expand((1, beam_width, self.hidden_size)), hiddens[1].expand((1, beam_width, self.hidden_size)))
        
        # start beam search
        for seq_id in range(self.max_dec_len):
            # calculate next input
            inputs = next_word_idx.unsqueeze(1) #(beam, 1)
            inputs = self.input_embedding(inputs) #(beam, 1, input_size)

            # calculate next hidden
            next_hidden_0 = torch.zeros(hiddens[0].size()).to(self.device)
            next_hidden_1 = torch.zeros(hiddens[1].size()).to(self.device)
            for beam_id in range(beam_width):
                next_hidden_0[0][beam_id][:] = hiddens[0][0][prev_beam_id[beam_id]][:]
                next_hidden_1[0][beam_id][:] = hiddens[1][0][prev_beam_id[beam_id]][:]

            # run through lstm
            outputs, hiddens = self.lstm(inputs, (next_hidden_0, next_hidden_1))
            outputs = self.output_fc(outputs.squeeze(1))
            scores  = -F.log_softmax(outputs, dim=1) # (beam, vocab_size)
            
            scores = curr_beam_scores.view(beam_width, 1).expand(beam_width, self.vocab_size) * scores

            # We have reached the max_dec_len, now back tracking
            if seq_id == self.max_dec_len-1:
                for beam_id in range(beam_width):
                    prediction_ids = []
                    next_beam_id = beam_id
                    for track_seq_id in range(seq_id, -1, -1):
                        track_word, track_beam_id = track_table[next_beam_id][track_seq_id]
                        prediction_ids.append(track_word)
                        next_beam_id = track_beam_id
                    prediction_ids.reverse()
                    hypos.append((scores[beam_id][2].cpu().data.tolist(), prediction_ids))
                break

            
            curr_beam_scores, topk_idx = scores.view(-1).topk(beam_width, largest=False) # (beam)
            
            next_word_idx = topk_idx % self.vocab_size
            prev_beam_id = topk_idx / self.vocab_size
            
            for beam_id in range(beam_width):
                track_table[beam_id].append((next_word_idx[beam_id].cpu().data.tolist(), prev_beam_id[beam_id].cpu().data.tolist()))

                # back tracking, if we meet end
                if next_word_idx[beam_id].cpu().data.tolist() == 2 or next_word_idx[beam_id].cpu().data.tolist() == 19:
                    prediction_ids = []
                    next_beam_id = beam_id
                    for track_seq_id in range(seq_id + 1, -1, -1):
                        track_word, track_beam_id = track_table[next_beam_id][track_seq_id]
                        prediction_ids.append(track_word)
                        next_beam_id = track_beam_id
                    prediction_ids.reverse()
                    hypos.append((curr_beam_scores[beam_id].cpu().data.tolist(), prediction_ids))
        hypos.sort(key=lambda x:x[0])
        return hypos[0][1]






