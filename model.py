import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src) #[64, 2, 1024], [1, 64, 512], [1, 64, 512]
        input = trg[:, 0] #[64]
        input_feed = None
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs, input_feed) #output:[64,16281], hidden:[2,64,512], cell:[2,64,512]
            # print(output.shape)
            outputs[:, t] = output #[64,16281]
            top1 = output.argmax(1) #[64]
            input = trg[:, t]
            input_feed=top1 
        return outputs