import torch
from torch import nn
from typing import List, Dict, Any
from src.data.components.config import PAD_token
import os
import json
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
from embeddings import GloveEmbedding, KazumaCharEmbedding

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, load_emb, fix_emb, n_layers=1):
        super(EncoderRNN, self).__init__()      
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size  
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        # self.domain_W = nn.Linear(hidden_size, nb_domain)

        if load_emb:
            with open(os.path.join("data/emb/", 'emb{}.json'.format(vocab_size))) as f:
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True
            print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

        if fix_emb:
            self.embedding.weight.requires_grad = False
    
    def get_state(self, bsz, device):
        # Get cell states and hidden states.
        return Variable(torch.zeros(2, bsz, self.hidden_size)).to(device)
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1), embedded.device)
        # hidden = Variable(torch.zeros(2, input_seqs.size(1), self.hidden_size))

        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)   
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs.transpose(0,1), hidden.unsqueeze(0)

class Generator(nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate, parallel_decode):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.lang = lang
        self.embedding = shared_emb 
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3*hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots
        self.parallel_decode = parallel_decode

        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches, use_teacher_forcing, slot_temp):
        device = encoded_hidden.device  # encoder hidden state と同じデバイスを取得

        all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size, device=device)
        all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate, device=device)

        """
        if USE_CUDA: 
            all_point_outputs = all_point_outputs.cuda()
            all_gate_outputs = all_gate_outputs.cuda()
        """
        
        # Get the slot embedding 
        slot_emb_dict = {}
        for i, slot in enumerate(slot_temp):
            # Domain embbeding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                domain_emb = self.Slot_emb(domain_w2idx.to(self.Slot_emb.weight.device))

            # Slot embbeding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx)
                # if USE_CUDA: slot_w2idx = slot_w2idx.cuda()
                slot_emb = self.Slot_emb(slot_w2idx.to(self.Slot_emb.weight.device))

            # Combine two embeddings as one query
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb
            slot_emb_exp = combined_emb.expand_as(encoded_hidden)
            if i == 0:
                slot_emb_arr = slot_emb_exp.clone()
            else:
                slot_emb_arr = torch.cat((slot_emb_arr, slot_emb_exp), dim=0)

        if self.parallel_decode:
            # Compute pointer-generator output, puting all (domain, slot) in one batch
            decoder_input = self.dropout_layer(slot_emb_arr).view(-1, self.hidden_size) # (batch*|slot|) * emb
            hidden = encoded_hidden.repeat(1, len(slot_temp), 1) # 1 * (batch*|slot|) * emb
            words_point_out = [[] for i in range(len(slot_temp))]
            words_class_out = []
            
            for wi in range(max_res_len):
                dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

                enc_out = encoded_outputs.repeat(len(slot_temp), 1, 1)
                enc_len = encoded_lens * len(slot_temp)
                context_vec, logits, prob = self.attend(enc_out, hidden.squeeze(0), enc_len)

                if wi == 0: 
                    all_gate_outputs = torch.reshape(self.W_gate(context_vec), all_gate_outputs.size())

                p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                p_context_ptr = torch.zeros(p_vocab.size(), device=p_vocab.device)
                p_context_ptr.scatter_add_(1, story.repeat(len(slot_temp), 1), prob)

                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)
                words = [self.lang.index2word[w_idx.item()] for w_idx in pred_word]
                
                for si in range(len(slot_temp)):
                    words_point_out[si].append(words[si*batch_size:(si+1)*batch_size])
                
                all_point_outputs[:, :, wi, :] = torch.reshape(final_p_vocab, (len(slot_temp), batch_size, self.vocab_size))
                
                if use_teacher_forcing:
                    decoder_input = self.embedding(torch.flatten(target_batches[:, :, wi].transpose(1,0)))
                else:
                    decoder_input = self.embedding(pred_word)   
                
                #  if USE_CUDA: decoder_input = decoder_input.cuda()
        else:
            # Compute pointer-generator output, decoding each (domain, slot) one-by-one
            words_point_out = []
            counter = 0
            for slot in slot_temp:
                hidden = encoded_hidden
                words = []
                slot_emb = slot_emb_dict[slot]
                decoder_input = self.dropout_layer(slot_emb).expand(batch_size, self.hidden_size)
                for wi in range(max_res_len):
                    dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)
                    context_vec, logits, prob = self.attend(encoded_outputs, hidden.squeeze(0), encoded_lens)
                    if wi == 0: 
                        all_gate_outputs[counter] = self.W_gate(context_vec)
                    p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                    p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                    vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                    p_context_ptr = torch.zeros(p_vocab.size(), device=p_vocab.device)
                    p_context_ptr.scatter_add_(1, story, prob)
                    final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                    vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                    pred_word = torch.argmax(final_p_vocab, dim=1)
                    words.append([self.lang.index2word[w_idx.item()] for w_idx in pred_word])
                    all_point_outputs[counter, :, wi, :] = final_p_vocab
                    if use_teacher_forcing:
                        decoder_input = self.embedding(target_batches[:, counter, wi]) # Chosen word is next input
                    else:
                        decoder_input = self.embedding(pred_word)   
                    # if USE_CUDA: decoder_input = decoder_input.cuda()
                counter += 1
                words_point_out.append(words)
        return all_point_outputs, all_gate_outputs, words_point_out, []

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        scores = F.softmax(scores_, dim=1)
        return scores



    
