import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from collections import Counter

class ArticleDataset(Dataset):
    def __init__(self, csv_file, classification_level="category", filter_params=None):
        """
        Args:
            csv_file (str): Chemin vers le fichier CSV.
            classification_level (str): "category" pour utiliser la partie avant le point,
                                        "sub_category" pour utiliser la partie après le point.
            filter_params (dict): Dictionnaire des critères de filtrage à appliquer (ex : {'min_papers': 10, 'min_freq': 3}).
        """
        # Chargement du CSV et suppression des doublons sur "summary"
        self.data = pd.read_csv(csv_file, engine="python").drop_duplicates(subset="summary")
        self.classification_level = classification_level
        
        # Création des colonnes pour la catégorie principale et la sous-catégorie
        self.data["cat_principale"] = self.data["category"].apply(lambda x: x.split(".")[0])
        self.data["sous_categorie"] = self.data["category"].apply(lambda x: x.split(".")[1] if len(x.split(".")) > 1 else None)
        
        # Construction initiale du vocabulaire sans filtrage de fréquence
        self.vocab = self.create_vocab(apply_filter=False)
        self.wtoi = {word: i for i, word in enumerate(self.vocab, start=1)}
        self.itow = {i: word for i, word in enumerate(self.vocab, start=1)}
        self.wtoi["<unk>"] = 0
        self.itow[0] = "<unk>"
        
        # Création du mapping pour les classes en fonction du niveau choisi
        self.set_classification_level(self.classification_level)
        
        # Application de filtres éventuels, notamment pour le vocabulaire
        if filter_params:
            self.apply_filters(filter_params)

    def create_vocab(self, apply_filter=False, min_freq=None):
        """
        Construit le vocabulaire à partir des titres.
        
        Args:
            apply_filter (bool): Si True, ne garde que les mots apparaissant plus de min_freq fois.
            min_freq (int): Seuil minimal pour le filtrage (utilisé uniquement si apply_filter est True).
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
        Met à jour le mapping des classes en fonction du niveau de classification choisi.
        """
        self.classification_level = level
        if level == "category":
            self.categories = list(self.data["cat_principale"].unique())
            self.ctoi = {cat: i for i, cat in enumerate(self.categories)}
        elif level == "sub_category":
            self.categories = list(self.data["sous_categorie"].unique())
            self.ctoi = {cat: i for i, cat in enumerate(self.categories)}
        else:
            raise ValueError("classification_level doit être 'category' ou 'sub_category'")
    
    def apply_filters(self, filter_params):
        """
        Applique des filtres sur self.data et met à jour les mappings.
        Exemples de filtres :
            - Filtrer pour ne garder que les classes ayant au moins 'min_papers' articles.
            - Garder uniquement les 'top_n' classes avec le plus d'articles.
            - Recalculer le vocabulaire en fonction d'un nouveau min_freq.
        Args:
            filter_params (dict): Dictionnaire de paramètres de filtrage.
                Exemples : {"min_papers": 10, "top_n": 5, "min_freq": 3}
        """
        # Exemple de filtre : mise à jour du vocabulaire avec min_freq si précisé
        if "min_freq" in filter_params:
            min_freq = filter_params["min_freq"]
            self.vocab = self.create_vocab(apply_filter=True, min_freq=min_freq)
            self.wtoi = {word: i for i, word in enumerate(self.vocab, start=1)}
            self.itow = {i: word for i, word in enumerate(self.vocab, start=1)}
            self.wtoi["<unk>"] = 0
            self.itow[0] = "<unk>"
        
        # Exemple de filtre : minimum de papiers par classe
        if "min_papers" in filter_params:
            if self.classification_level == "category":
                counts = self.data["cat_principale"].value_counts()
                valid = counts[counts >= filter_params["min_papers"]].index
                self.data = self.data[self.data["cat_principale"].isin(valid)]
            elif self.classification_level == "sub_category":
                counts = self.data["sous_categorie"].value_counts()
                valid = counts[counts >= filter_params["min_papers"]].index
                self.data = self.data[self.data["sous_categorie"].isin(valid)]
        
        # Exemple de filtre : ne garder que les top n classes
        if "top_n" in filter_params:
            if self.classification_level == "category":
                counts = self.data["cat_principale"].value_counts()
                top_classes = counts.nlargest(filter_params["top_n"]).index
                self.data = self.data[self.data["cat_principale"].isin(top_classes)]
            elif self.classification_level == "sub_category":
                counts = self.data["sous_categorie"].value_counts()
                top_classes = counts.nlargest(filter_params["top_n"]).index
                self.data = self.data[self.data["sous_categorie"].isin(top_classes)]
        
        # Recalcule le mapping des classes selon le niveau de classification choisi
        self.set_classification_level(self.classification_level)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["title"]
        # Choix de la classe en fonction du niveau de classification
        if self.classification_level == "category":
            cat = row["cat_principale"]
        elif self.classification_level == "sub_category":
            cat = row["sous_categorie"]
        
        words = re.findall(r'[a-z0-9]+', text.lower())
        
        indices = torch.tensor(
            [self.wtoi.get(word, self.wtoi["<unk>"]) for word in words],
            dtype=torch.long
        )
        label = torch.tensor(self.ctoi[cat], dtype=torch.long)
        return indices, label
