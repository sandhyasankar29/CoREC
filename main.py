import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import BahdanauAttention, Decoder
from model import Seq2Seq
from train import train
from evaluate import evaluate
from dataset import DiffMsgDataset, create_collate_fn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 1000
SRC_LENGTH_TRUNC = 100
TGT_LENGTH_TRUNC = 30
EMBEDDING_DIM = 512
ENC_LAYERS = 2
DEC_LAYERS = 2
HIDDEN_DIM = 512
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT = 0.1
TRAIN_STEPS = 100000
LOWER = True
MIN_FREQ = 2
CLIP=1
TOTAL = 22112

if __name__ == '__main__':
    train_diff_file = 'data/top1000/cleaned.train.diff'
    train_commit_file = 'data/top1000/cleaned.train.msg'
    valid_diff_file = 'data/top1000/cleaned.valid.diff'
    valid_commit_file = 'data/top1000/cleaned.valid.msg'
    test_diff_file = 'data/top1000/cleaned.test.diff'
    test_commit_file = 'data/top1000/cleaned.test.msg'
    train_dataset = DiffMsgDataset(train_diff_file, train_commit_file, SRC_LENGTH_TRUNC, TGT_LENGTH_TRUNC)
    valid_dataset = DiffMsgDataset(valid_diff_file, valid_commit_file, SRC_LENGTH_TRUNC, TGT_LENGTH_TRUNC)
    test_dataset = DiffMsgDataset(test_diff_file, test_commit_file, SRC_LENGTH_TRUNC, TGT_LENGTH_TRUNC)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=create_collate_fn(SRC_LENGTH_TRUNC, TGT_LENGTH_TRUNC), shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=create_collate_fn(SRC_LENGTH_TRUNC, TGT_LENGTH_TRUNC), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=create_collate_fn(SRC_LENGTH_TRUNC, TGT_LENGTH_TRUNC), shuffle=False)
    
    encoder = Encoder(len(train_dataset.src_vocab), EMBEDDING_DIM, HIDDEN_DIM)
    attention = BahdanauAttention(HIDDEN_DIM, HIDDEN_DIM)
    decoder = Decoder(len(train_dataset.tgt_vocab), EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM, attention, TOTAL, BATCH_SIZE)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
# Set up training
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Training loop
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    print(f"Epoch: {epoch} Train_loss: {train_loss}")
    valid_loss = evaluate(model, valid_loader, criterion)
    print(f"Epoch: {epoch} Train_loss: {train_loss} Valid_loss: {valid_loss}")