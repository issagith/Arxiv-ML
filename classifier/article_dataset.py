import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from collections import Counter

class ArticleDataset(Dataset):
    def __init__(self, csv_file, classification_level="category", filter_params=None):
        # Load the CSV and remove duplicates based on "summary"
        self.data = pd.read_csv(csv_file, engine="python").drop_duplicates(subset="summary")
        self.classification_level = classification_level
        
        # Create columns for the main category and sub-category
        self.data["main_category"] = self.data["category"].apply(lambda x: x.split(".")[0])
        self.data["sub_category"] = self.data["category"].apply(lambda x: x.split(".")[1] if len(x.split(".")) > 1 else None)
        
        # Initial construction of the vocabulary without frequency filtering
        self.vocab = self.create_vocab(apply_filter=False)
        self.wtoi = {word: i for i, word in enumerate(self.vocab, start=1)}
        self.itow = {i: word for i, word in enumerate(self.vocab, start=1)}
        self.wtoi["<unk>"] = 0
        self.itow[0] = "<unk>"
        
        # Create the mapping for classes based on the chosen level
        self.set_classification_level(self.classification_level)
        
        # Apply any filters, especially for the vocabulary
        if filter_params:
            self.apply_filters(filter_params)

    def create_vocab(self, apply_filter=False, min_freq=None):
        vocab_counter = Counter()
        for title in self.data["title"]:
            words = re.findall(r'[a-z0-9]+', title.lower())
            vocab_counter.update(words)
        if apply_filter and min_freq is not None:
            return [word for word, count in vocab_counter.items() if count > min_freq]
        else:
            return list(vocab_counter.keys())
    
    def set_classification_level(self, level):
        self.classification_level = level
        if level == "category":
            self.categories = list(self.data["main_category"].unique())
            self.ctoi = {cat: i for i, cat in enumerate(self.categories)}
        elif level == "sub_category":
            self.categories = list(self.data["sub_category"].unique())
            self.ctoi = {cat: i for i, cat in enumerate(self.categories)}
        else:
            raise ValueError("classification_level must be 'category' or 'sub_category'")
    
    def apply_filters(self, filter_params):
        """
        Apply filters on self.data and update the mappings.
        Args:
            filter_params (dict): Dictionary of filtering parameters.
                Examples: {"min_papers": 10, "top_n": 5, "min_freq": 3}
        """
        
        if "min_freq" in filter_params:
            min_freq = filter_params["min_freq"]
            self.vocab = self.create_vocab(apply_filter=True, min_freq=min_freq)
            self.wtoi = {word: i for i, word in enumerate(self.vocab, start=1)}
            self.itow = {i: word for i, word in enumerate(self.vocab, start=1)}
            self.wtoi["<unk>"] = 0
            self.itow[0] = "<unk>"
        
        if "min_papers" in filter_params:
            if self.classification_level == "category":
                counts = self.data["main_category"].value_counts()
                valid = counts[counts >= filter_params["min_papers"]].index
                self.data = self.data[self.data["main_category"].isin(valid)]
            elif self.classification_level == "sub_category":
                counts = self.data["sub_category"].value_counts()
                valid = counts[counts >= filter_params["min_papers"]].index
                self.data = self.data[self.data["sub_category"].isin(valid)]
        
        if "top_n" in filter_params:
            if self.classification_level == "category":
                counts = self.data["main_category"].value_counts()
                top_classes = counts.nlargest(filter_params["top_n"]).index
                self.data = self.data[self.data["main_category"].isin(top_classes)]
            elif self.classification_level == "sub_category":
                counts = self.data["sub_category"].value_counts()
                top_classes = counts.nlargest(filter_params["top_n"]).index
                self.data = self.data[self.data["sub_category"].isin(top_classes)]
        
        # Recalculate the class mapping based on the chosen classification level
        self.set_classification_level(self.classification_level)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["title"]
        # Choose the class based on the classification level
        if self.classification_level == "category":
            cat = row["main_category"]
        elif self.classification_level == "sub_category":
            cat = row["sub_category"]
        
        words = re.findall(r'[a-z0-9]+', text.lower())
        
        indices = torch.tensor(
            [self.wtoi.get(word, self.wtoi["<unk>"]) for word in words],
            dtype=torch.long
        )
        label = torch.tensor(self.ctoi[cat], dtype=torch.long)
        return indices, label
