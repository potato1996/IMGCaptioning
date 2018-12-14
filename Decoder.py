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
                    inputs = torch.tensor([1], dtype=torch.long).cuda()
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
        seq = torch.LongTensor(self.max_dec_len).zero_()
        seqLogprobs = torch.FloatTensor(self.max_dec_len)

        # 1. the first input should be image
        inputs = img_embedding.expand(beam_width, self.input_size).unsequeeze(1)  # (beam_width, 1, input_size)

        # 2. run the first input through the lstm
        _, hiddens = self.lstm(inputs)

        # New
        batch_size = 1

        assert beam_width <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.max_dec_len, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.max_dec_len, batch_size)
        # lets process every image independently for now, for simplicity


        it = torch.ones(beam_width)  # bean_width
        xt = self.input_embedding(it)  # (beam_width, input_size)

        output, state = self.lstm(xt.unsqueeze(0), hiddens)
        logprobs = F.log_softmax(self.output_fc(self.dropout(output.squeeze(0))))

        self.done_beams = self.beam_search(state, logprobs, beam_width)
        seq[:] = self.done_beams[0]['seq']  # the first beam has highest cumulative score
        seqLogprobs[:] = self.done_beams[0]['logps']

        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def beam_search(self, state, logprobs, beam_size=10):
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # INPUTS:
            # logprobsf: probabilities augmented after diversity
            # beam_size: obvious
            # t        : time instant
            # beam_seq : tensor contanining the beams
            # beam_seq_logprobs: tensor contanining the beam logprobs
            # beam_logprobs_sum: tensor contanining joint logprobs
            # OUPUTS:
            # beam_seq : tensor containing the word indices of the decoded captions
            # beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            # beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):  # for each column (word, essentially)
                for q in range(rows):  # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob.cpu()
                    candidates.append(dict(c=ix[q, c], q=q,
                                           p=candidate_logprob,
                                           r=local_logprob))
            candidates = sorted(candidates, key=lambda x: -x['p'])

            new_state = [_.clone() for _ in state]
            # beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
                # we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                # fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # rearrange recurrent states
                for state_ix in range(len(new_state)):
                    #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]  # dimension one is time step
                # append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c']  # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
                beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        # start beam search

        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        # running sum of logprobs for each beam
        beam_logprobs_sum = torch.zeros(beam_size)
        done_beams = []

        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            logprobsf = logprobs.data.float()  # lets go to CPU for more efficiency in indexing operations
            # suppress UNK tokens in the decoding
            logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000

            beam_seq, \
            beam_seq_logprobs, \
            beam_logprobs_sum, \
            state, \
            candidates_divm = beam_step(logprobsf,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum,
                                        state)

            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # encode as vectors
            it = beam_seq[t]
            logprobs, state = self.get_logprobs_state(Variable(it.cuda()), *(args + (state,)))

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams
