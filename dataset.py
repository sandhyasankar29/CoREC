import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# 1. Data Preprocessing
class DiffMsgDataset(Dataset):
    def __init__(self, diff_file, msg_file, src_length_trunc, tgt_length_trunc):
        with open(diff_file, 'r') as f:
            diffs = f.readlines()
        with open(msg_file, 'r') as f:
            msgs = f.readlines()

        # Remove entries where either diff or msg is empty
        self.diffs, self.msgs = [], []
        for diff, msg in zip(diffs, msgs):
            if diff.strip() and msg.strip():
                self.diffs.append(diff.strip().split()[:src_length_trunc])
                self.msgs.append(msg.strip().split()[:tgt_length_trunc])

        self.src_length_trunc = src_length_trunc
        self.tgt_length_trunc = tgt_length_trunc

        # Build vocabularies
        self.src_vocab = self.build_vocab(self.diffs)
        self.tgt_vocab = self.build_vocab(self.msgs)

    def build_vocab(self, sentences):
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab

    def __len__(self):
        return len(self.diffs)

    def __getitem__(self, idx):
        diff = [self.src_vocab.get(word, self.src_vocab['<unk>']) for word in self.diffs[idx]]
        msg = [self.tgt_vocab['<sos>']] + [self.tgt_vocab.get(word, self.tgt_vocab['<unk>']) for word in self.msgs[idx]] + [self.tgt_vocab['<eos>']]
        return torch.tensor(diff), torch.tensor(msg)
def create_collate_fn(src_max_len, tgt_max_len):
    def collate_fn(batch):
        diffs, msgs = zip(*batch)
        
        # Padding for source sequences
        padded_diffs = []
        for diff in diffs:
            diff_len = len(diff)
            if diff_len < src_max_len:
                padding = torch.zeros(src_max_len - diff_len, dtype=torch.long)
                padded_diff = torch.cat([diff, padding])
            else:
                padded_diff = diff[:src_max_len]
            padded_diffs.append(padded_diff)
        
        # Padding for target sequences
        padded_msgs = []
        for msg in msgs:
            msg_len = len(msg)
            if msg_len < tgt_max_len:
                padding = torch.zeros(tgt_max_len - msg_len, dtype=torch.long)
                padded_msg = torch.cat([msg, padding])
            else:
                padded_msg = msg[:tgt_max_len]
            padded_msgs.append(padded_msg)

        return torch.stack(padded_diffs), torch.stack(padded_msgs)
    return collate_fn