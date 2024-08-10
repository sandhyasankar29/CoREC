import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embeding(x)*math.sqrt(self.d_model)

class Attention(nn.Module):
    def __init__(self, n_hidden_enc: int, n_hidden_dec: int) -> None:
        super().__init__()
        self.n_hidden_enc = n_hidden_enc
        self.n_hidden_dec = n_hidden_dec
        self.W = nn.Linear(2*n_hidden_enc + n_hidden_dec, n_hidden_dec, bias=False)
        self.V = nn.Parameter(torch.rand(n_hidden_dec))
    
    def forward(self, hidden_dec, last_layer_enc):
        """
            PARAMS:
                hidden_dec: [b, n_layers, n_hidden_dec]     (1st hidden_dec = encoder's last_h's last layer)
                last_layer_enc: [b, seq_len, n_hidden_enc*2]
            
            RETURN:
                attn_weights: [b, src_seq_len]
        """
        batch_size = last_layer_enc.size(0)
        src_seq_len = last_layer_enc.size(1)

        hidden_dec = hidden_dec[:, -1, :].unsqueeze(1).repeat(1, src_seq_len, 1)        #[b, src_seq_len, n_hidden_dec]
        tanh_W_s_h = torch.tanh(self.W(torch.cat((hidden_dec,last_layer_enc), dim=2)))  #[b, src_seq_len, n_hidden_dec]
        tanh_W_s_h = tanh_W_s_h.permute(0,2,1)  #[b, n_hidden_dec, src_seq_len]
        V = self.V.repeat(batch_size, 1).unsqueeze(1) #[b, 1, n_hidden_dec]
        e = torch.bmm(V, tanh_W_s_h).squeeze(1) #[b, seq_len]
        attn_weights = F.softmax(e, dim=1) #[b, seq_len]
        return attn_weights

