#!/bin/bash

# Créer le dossier data s'il n'existe pas
mkdir -p data

# Télécharger le dataset si ce n'est pas déjà fait
if [ ! -f "data/articles.csv" ]; then
    echo "Téléchargement du dataset..."
    python -c "
from datasets import load_dataset
import pandas as pd

# Charger le dataset depuis Hugging Face
dataset = load_dataset('issaHF/arxiv-ml-dataset', split='train')

# Convertir en DataFrame et sauvegarder en CSV
df = pd.DataFrame(dataset)
df.to_csv('data/articles.csv', index=False)
"
fi