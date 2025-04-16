
import json
import torch
from torch.utils.data import Dataset
import sentencepiece as spm

class BPEWrapper:
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)

    def eos_id(self):
        return self.sp.eos_id()

    def pad_id(self):
        return self.sp.pad_id()

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.samples = []
        for item in data:
            tokens = tokenizer.encode(item['prompt'] + ' ' + item['completion'])
            tokens = tokens[:max_len]
            self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        return tokens[:-1], tokens[1:]
