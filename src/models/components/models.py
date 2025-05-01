from src.models.components.components import *
from random import random
from src.utils.utils import masked_cross_entropy_for_value

class TRADE(nn.Module):
    def __init__(self, hidden_size, lang, path, dropout, slots, genSample, unk_mask, teacher_forcing_ratio, use_gate, load_emb, fix_emb, parallel_decode, nb_train_vocab=0):
        super(TRADE, self).__init__()
        self.name = "TRADE"
        self.hidden_size = hidden_size    
        self.lang = lang[0]
        self.mem_lang = lang[1] 
        self.dropout = dropout
        self.slots = slots[0]
        self.slot_temp = slots[2]
        self.gating_dict = {"ptr":0, "dontcare":1, "none":2}
        self.inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
        self.nb_gate = len(self.gating_dict)
        self.cross_entorpy = nn.CrossEntropyLoss()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_gate = use_gate
        self.unk_mask = unk_mask
        self.genSample = genSample
        self.load_emb = load_emb
        self.fix_emb = fix_emb
        self.parallel_decode = parallel_decode

        self.encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout, self.load_emb, self.fix_emb)
        self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout, self.slots, self.nb_gate, parallel_decode) 
        
        if path:            
            print("MODEL {} LOADED".format(str(path)))
            trained_encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
            trained_decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
            
            self.encoder.load_state_dict(trained_encoder.state_dict())
            self.decoder.load_state_dict(trained_decoder.state_dict())

        self.reset()

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        use_teacher_forcing = self.training and random() < self.teacher_forcing_ratio

        all_point_outputs, gates, words_point_out, words_class_out = self.encode_and_decode(
            batch, use_teacher_forcing, self.slot_temp
        )
        return all_point_outputs, gates, words_point_out, words_class_out

    def loss(self, all_point_outputs, gates, words_point_out, words_class_out, batch):
        loss_ptr = masked_cross_entropy_for_value(
                all_point_outputs.transpose(0, 1).contiguous(),
                batch["generate_y"].contiguous(),
                batch["y_lengths"]
            )
        
        gates = gates.to(batch["gating_label"].device)
        loss_gate = self.cross_entorpy(
            gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)),
            batch["gating_label"].contiguous().view(-1)
        )
        loss = loss_ptr + loss_gate if self.model.use_gate else loss_ptr
        return loss
    
    def encode_and_decode(self, data, use_teacher_forcing, slot_temp):
        # Build unknown mask for memory to encourage generalization
        story = data['context']
        if self.unk_mask and self.decoder.training:
            story_size = story.size()
            rand_mask = np.ones(story_size, dtype=np.float32)
            bi_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))], 1-self.dropout)[0]
            rand_mask = rand_mask * bi_mask
            rand_mask = torch.from_numpy(rand_mask).to(dtype=story.dtype, device=story.device)
            story = story * rand_mask.long()

        # Encode dialog history
        encoded_outputs, encoded_hidden = self.encoder(story.transpose(0, 1), data['context_len'])

        # Get the words that can be copy from the memory
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10
        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size, \
            encoded_hidden, encoded_outputs, data['context_len'], story, max_res_len, data['generate_y'], \
            use_teacher_forcing, slot_temp) 
        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0
