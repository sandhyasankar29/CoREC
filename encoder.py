import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)  
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # Adjusting for bidirection

    def forward(self, src):
        embedded = self.embedding(src) #[Batch_size, src_len, embedding_dim] -> [64, 100, 512]
        outputs, (hidden, cell) = self.rnn(embedded) #output:[Batch_size, Layers, Directions*hidden_dim] -> [64, 2, 1024]
        #hidden:[Directions*Layers, Batch_size, hidden_dim] -> [4, 64, 512]
        #cell: [Directions*Layers, Batch_size, hidden_dim] -> [4, 64, 512]
        
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))) #fc_layer:[64, 1024]->tanh:[64, 512]
        cell = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1))) #fc_layer:[64, 1024]->tanh:[64, 512]
        
        return outputs, hidden.unsqueeze(0), cell.unsqueeze(0) #[64, 2, 1024], [1, 64, 512], [1, 64, 512]
