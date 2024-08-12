import random
import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.attn = nn.Linear(enc_hidden_dim*2 + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Parameter(torch.rand(dec_hidden_dim))

    def forward(self, hidden, encoder_outputs):
        '''hidden: [64, 512], encoder_outputs:[64, 100, 1024]'''
        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1) #hidden: [64, 100, 512]
        # print(hidden.shape)
        # print(encoder_outputs.shape)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) #energy: [64,100,1536] -> [64,100,512]
        energy = energy.permute(0, 2, 1) #energy: [64, 512, 100]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1) #[64, 1, 512]
        attention = torch.bmm(v, energy).squeeze(1) #[64,1,512]X[64,512,100]->[64,1,100] (after squeeze) [64,100]
        return torch.softmax(attention, dim=1) #[64,100]

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, enc_hidden_dim, dec_hidden_dim, attention, total, batch_size):
        super(Decoder, self).__init__()
        self.output_dim = output_dim #16281
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim + 2*enc_hidden_dim, dec_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(2*enc_hidden_dim + dec_hidden_dim + embedding_dim, output_dim)
        self.attention = attention
        
        # Initialize step and test attributes
        self.step = 0
        self.test = 0
        self.total = total  # Total number of training samples
        self.batch_size = batch_size  # Batch size for training

    def randn_choose(self, emb_t, input_feed):
        if self.test == 1 or input_feed == None:
            return emb_t
        total = self.total * 4
        steps_epoch = total // self.batch_size
        epoch = float(self.step // steps_epoch)
        x = random.uniform(0, 1)
        mu = 12.0
        p = mu / (mu + torch.exp(torch.tensor(epoch / mu)))
        if x > p:
            return input_feed
        else:
            return emb_t

    def forward(self, input, hidden, cell, encoder_outputs, input_feed=None):
        '''hidden: [1, 64, 512], cell: [1, 64, 512] encoder_outputs: [64, 100, 1024]'''
        input = self.randn_choose(input, input_feed)
        input = input.unsqueeze(1) #[Batch_size,1] -> [64,1]
        embedded = self.embedding(input) #[Batch_size, embed_dim] -> [64, 1, 512]
        attention_weights = self.attention(hidden[-1], encoder_outputs) # attention([64, 512], [64, 100, 512]) -> [64,100]
        attention_weights = attention_weights.unsqueeze(1) #[64, 1, 100]
        context = torch.bmm(attention_weights, encoder_outputs) #[64,1,100]X[64,100,1024] -> [64,1,1024]
        rnn_input = torch.cat((embedded, context), dim=2) #[64,1,1536]

        output, (hidden, cell) = self.rnn(rnn_input) #output:[Batch_size, seq_len, Directions*hidden_dim] -> [64, 1, 512]
        #hidden:[Directions*Layers, Batch_size, hidden_dim] -> [2, 64, 512]
        #cell: [Directions*Layers, Batch_size, hidden_dim] -> [2, 64, 512]
        output = torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1) #[64, 2048]
        prediction = self.fc_out(output) #[64, 16281]
        prediction = torch.softmax(prediction, dim=1)
        
        # Update step during each forward pass
        self.step += 1

        return prediction, hidden, cell #prediction:[64,16281], hidden:[1,64,512], cell:[1,64,512]
