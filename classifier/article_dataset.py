# article_dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from collections import Counter
from categories import subcats, cats

class ArticleDataset(Dataset):
    def __init__(self, csv_file, classification_level="category", filter_params=None):
        """
        Initializes the dataset from a CSV file.
        Constructs vocabulary and class mappings.
        """
        # Load CSV and drop duplicates based on "summary"
        self.data = pd.read_csv(csv_file, engine="python").drop_duplicates(subset="summary")
        self.classification_level = classification_level
        
        # Create columns for main and sub categories
        self.data["main_category"] = self.data["category"].apply(lambda x: x.split(".")[0])
        self.data["sub_category"] = self.data["category"].apply(lambda x: x.split(".")[1] if len(x.split(".")) > 1 else None)

        # filter out rows with unvalid categories
        self.data = self.data[self.data["main_category"].isin(cats)]

        # filter out rows with unvalid subcategories
        def valid_sub(row):
            main = row["main_category"]
            subs = subcats.get(main, [])
            # aucun sous‐cat défini → on n’accepte que les entrées 'main' pures
            if not subs:
                return row["sub_category"] is None
            # sinon, on vérifie que 'main.sub' est dans la liste
            full = f"{main}.{row['sub_category']}"
            return full in subs
        
        self.data = (self.data.loc[self.data.apply(valid_sub, axis=1)].reset_index(drop=True))

        # Build vocabulary without frequency filtering
        self.vocab = self.create_vocab(apply_filter=False)
        self.word_to_index = {word: i for i, word in enumerate(self.vocab, start=1)}
        self.index_to_word = {i: word for i, word in enumerate(self.vocab, start=1)}
        self.word_to_index["<unk>"] = 0
        self.index_to_word[0] = "<unk>"
        
        # Create class mappings based on chosen classification level
        self.set_classification_level(self.classification_level)
        
        # Apply filters if provided
        if filter_params:
            self.apply_filters(filter_params)

    def create_vocab(self, apply_filter=False, min_freq=None):
        """
        Creates the vocabulary from article titles.
        If apply_filter is True, only words with frequency > min_freq are kept.
        """
        vocab_counter = Counter()
        for title in self.data["title"]:
            words = re.findall(r'[a-z0-9]+', title.lower())
            vocab_counter.update(words)
        if apply_filter and min_freq is not None:
            return [word for word, count in vocab_counter.items() if count > min_freq]
        else:
            return list(vocab_counter.keys())
    
    def set_classification_level(self, level):
        """
        Sets the classification level ('category' or 'sub_category')
        and builds the corresponding class mappings.
        """
        self.classification_level = level
        if level == "category":
            self.categories = list(self.data["main_category"].unique())
        elif level == "sub_category":
            self.categories = list(self.data["sub_category"].unique())
        else:
            raise ValueError("classification_level must be 'category' or 'sub_category'")
        self.class_to_index = {cat: i for i, cat in enumerate(self.categories)}
        self.index_to_class = {i: cat for i, cat in enumerate(self.categories)}
    
    def apply_filters(self, filter_params):
        """
        Applies filters on the dataset and updates mappings.
        Example filter_params: {"min_papers": 10, "top_n": 5, "min_freq": 3}
        """
        if "min_freq" in filter_params:
            min_freq = filter_params["min_freq"]
            self.vocab = self.create_vocab(apply_filter=True, min_freq=min_freq)
            self.word_to_index = {word: i for i, word in enumerate(self.vocab, start=1)}
            self.index_to_word = {i: word for i, word in enumerate(self.vocab, start=1)}
            self.word_to_index["<unk>"] = 0
            self.index_to_word[0] = "<unk>"
        
        if "min_papers" in filter_params:
            if self.classification_level == "category":
                counts = self.data["main_category"].value_counts()
                valid = counts[counts >= filter_params["min_papers"]].index
                self.data = self.data[self.data["main_category"].isin(valid)]
            elif self.classification_level == "sub_category":
                counts = self.data["sub_category"].value_counts()
                valid = counts[counts >= filter_params["min_papers"]].index
                self.data = self.data[self.data["sub_category"].isin(valid)]

        if "max_papers" in filter_params:
            max_p = filter_params["max_papers"]
            if self.classification_level == "category":
                grp = self.data.groupby("main_category", group_keys=False)
            else:  # sub_category
                grp = self.data.groupby("sub_category", group_keys=False)
            # pour chaque groupe, on prend un échantillon de taille <= max_p
            self.data = grp.apply(lambda df: df.sample(n=min(len(df), max_p))).reset_index(drop=True)
        
        if "top_n" in filter_params:
            if self.classification_level == "category":
                counts = self.data["main_category"].value_counts()
                top_classes = counts.nlargest(filter_params["top_n"]).index
                self.data = self.data[self.data["main_category"].isin(top_classes)]
            elif self.classification_level == "sub_category":
                counts = self.data["sub_category"].value_counts()
                top_classes = counts.nlargest(filter_params["top_n"]).index
                self.data = self.data[self.data["sub_category"].isin(top_classes)]
        
        # Recalculate the class mappings after filtering
        self.set_classification_level(self.classification_level)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns the tokenized title and the corresponding class label.
        """
        row = self.data.iloc[idx]
        text = row["title"]
        if self.classification_level == "category":
            cat = row["main_category"]
        elif self.classification_level == "sub_category":
            cat = row["sub_category"]
        
        words = re.findall(r'[a-z0-9]+', text.lower())
        indices = torch.tensor([self.word_to_index.get(word, self.word_to_index["<unk>"]) for word in words],
                               dtype=torch.long)
        label = torch.tensor(self.class_to_index[cat], dtype=torch.long)
        return indices, label


if __name__ == "__main__":
    # Example usage
    dataset = ArticleDataset("data/articles.csv", classification_level="category")
    
    