import random
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
#         print ("Encoder", output.shape)
#         print ("Encoder", hidden.shape)
#         print ("Encoder", cell.shape)
        
        return output, hidden, cell

#       Concat attention
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, device):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.device      = device
        self.activation  = nn.Tanh()
        
        self.Wa = nn.Linear(
                      in_features= (enc_hid_dim +
                                    dec_hid_dim),
                      out_features=(enc_hid_dim +
                                    dec_hid_dim))
        
        self.Va = nn.Linear(
                      in_features=enc_hid_dim + dec_hid_dim,
                      out_features=1)
        self.scalar_product = nn.Sequential(
                                  self.Wa,
                                  self.activation,
                                  self.Va)
         
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, hidden, encoder_outputs):
#         print ("Attention, hidden_shape :", hidden.shape)
#         print ("Attention, enc_outputs  :", encoder_outputs.shape)
        assert hidden.shape[0] == encoder_outputs.shape[1]
        
        att_weights = torch.zeros(
                          encoder_outputs.shape[0],
                          hidden.shape[0],
                          dtype=torch.float,
                          device=self.device)
        
        for i in range(encoder_outputs.shape[0]):
            att_weights[i, :] = self.scalar_product(
                                    torch.cat(
                                        [hidden,
                                         encoder_outputs[i, :, :]],
                                        dim=1)).squeeze(1)

        att_weights = self.softmax(att_weights)
        
#         print ("Attention, weights :", att_weights.unsqueeze(2).shape)
#         print ("Attention, enc_out :", encoder_outputs.shape)
        
        return (att_weights.unsqueeze(2) * encoder_outputs).sum(dim=0)
    
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTMCell(input_size =emb_dim,
                               hidden_size=dec_hid_dim)
        # <YOUR CODE HERE>
        
        self.out = nn.Linear(in_features = (enc_hid_dim +
                                            dec_hid_dim),
                             out_features= output_dim) # might be wrong
        # <YOUR CODE HERE>
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
#         print ("Decoder, input  :", input.shape)
#         print ("Decoder, hidden :", hidden.shape)
#         print ("Decoder, cell   :", cell.shape)
#         print ("Decoder, encod_outputs.shape", encoder_outputs.shape)
        
        input = self.embedding(input)
#         print ("Decoder, input_2", input.shape)
#         print ("Decoder, hidden ", hidden.shape)
#         print ("Decoder, cell   ", cell.shape)
        
        
        # Calculating next state
        hid_next, cell_next = self.rnn(input, (hidden.squeeze(0), cell.squeeze(0)))
#         hid_next, cell_next = self.rnn(input, (hidden, cell))
#         print ("Decoder hid and cell :", hid_next.shape, cell_next.shape)
        
        
        # Attention
#         print ("hid_next device is", hid_next.device)
#         print ("enc_outs device is", encoder_outputs.device)
        
        attention_output = self.attention(hid_next,
                                          encoder_outputs)
#         print ("Decoder, attention output :", attention_output.shape)
        output = self.out(
                     torch.cat(
                         [hid_next,
                          attention_output],
                         dim=1))
#         print ("Decoder, output :", output.shape)
        
        
        
        return output, (hid_next, cell_next)# (output, (hidden, cell)) must be returned
        # <YOUR CODE HERE>
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device  = device
#         print ("My device is ", self.device)
        
        assert encoder.hid_dim == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
#         print ("Seq2Seq  trg", trg.shape)
        input = trg[0,:]
#         print ("Seq2Seq  input", input.shape)
        
        for t in range(1, max_len):
#             print ("Seq2Seq input" , input.shape)
#             print ("Seq2Seq hidden", hidden.shape)
#             print ("Seq2Seq cell"  , cell.shape)
            
            output, (hidden, cell) = self.decoder(input, hidden, cell, enc_states)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs