import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from collections import Counter

class ArticleDataset(Dataset):
    def __init__(self, csv_file, min_freq=5):
        self.data = pd.read_csv(csv_file, engine="python").drop_duplicates(subset="summary")
        self.min_freq = min_freq
        self.vocab = self.create_vocab()
        self.categories = list(self.data.category.unique())
        self.wtoi = {word: i for i, word in enumerate(self.vocab, start=1)}
        self.itow = {i: word for i, word in enumerate(self.vocab, start=1)}
        self.wtoi["<unk>"] = 0
        self.itow[0] = "<unk>"
        self.ctoi = {cat: i for i, cat in enumerate(self.categories)}
        
    def create_vocab(self):
        vocab_counter = Counter()
        for title in self.data["title"]:
            words = re.findall(r'[a-z0-9]+', title.lower())
            vocab_counter.update(words)
        filtered_vocab = [word for word, count in vocab_counter.items() if count > self.min_freq]
        return filtered_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["title"]
        category = row["category"]
       
        words = re.findall(r'[a-z0-9]+', text.lower())
        
        indices = torch.tensor(
            [self.wtoi.get(word, self.wtoi["<unk>"]) for word in words],
            dtype=torch.long
        )
        label = torch.tensor(self.ctoi[category], dtype=torch.long)
        return indices, label