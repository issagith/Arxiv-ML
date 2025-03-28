import os
import torch
import numpy as np
from article_dataset import ArticleDataset
from models.mlp_classifier import MLPClassifier
from train import train 
from eval import evaluate_model, plot_confusion_matrix, analyze_errors
from gensim.models import KeyedVectors

# -----------------------------
# Constantes et paramètres généraux
CSV_FILE = "data/articles.csv"
CLASSIFICATION_LEVEL = "category"  # "category" ou "sub_category"

# Hyperparamètres du modèle
EMBEDDING_DIM = 64
HIDDEN_DIM = 512
NUM_HIDDEN_LAYERS = 1
IS_CUSTOM_EMB = False
FREEZE_EMBEDDINGS = False

# -----------------------------
# Initialisation du dataset
print("Initialisation du dataset...")
dataset = ArticleDataset(CSV_FILE, CLASSIFICATION_LEVEL)

# Application des filtres sur le dataset
filters = {
    "min_papers": 5000,  # nombre minimum de documents par catégorie
    "min_freq": 2,       # fréquence minimale d'apparition d'un mot pour être inclus dans le vocabulaire
}
dataset.apply_filters(filters)
print("Dataset prêt !")

# Détermination des dimensions
VOCAB_SIZE = len(dataset.wtoi)
NUM_CLASSES = len(dataset.ctoi)

hyperparams = {
    'vocab_size': VOCAB_SIZE,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': HIDDEN_DIM,
    'num_classes': NUM_CLASSES,
    'num_hidden_layers': NUM_HIDDEN_LAYERS,
    'is_custom_emb': IS_CUSTOM_EMB,
    'freeze_embeddings': FREEZE_EMBEDDINGS
}

# Choix de l'appareil (GPU si disponible)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Chargement des embeddings Word2Vec depuis le fichier local
if not IS_CUSTOM_EMB:
    w2v_path = os.path.join("data", "word2vec-google-news-300.bin")
    if os.path.exists(w2v_path):
        print("Chargement du modèle Word2Vec à partir du fichier local...")
        w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    else:
        raise FileNotFoundError(f"Le fichier {w2v_path} est introuvable. Veuillez vérifier qu'il est présent.")

    # Construction de la matrice d'embeddings (réduction de dimension de 300 à EMBEDDING_DIM)
    embedding_weights = torch.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    missing_words = 0
    for word, idx in dataset.wtoi.items():
        if word in w2v_model:
            embedding_weights[idx] = torch.tensor(w2v_model[word])[:EMBEDDING_DIM]
        else:
            missing_words += 1
            # Initialisation aléatoire pour les mots absents
            embedding_weights[idx] = torch.tensor(np.random.randn(EMBEDDING_DIM), dtype=torch.float)
    print(f"{missing_words}/{len(dataset.wtoi)} mots ne sont pas présents dans Word2Vec.")

    # Création du modèle avec les embeddings pré-entraînés
    model = MLPClassifier(
        VOCAB_SIZE,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_CLASSES,
        NUM_HIDDEN_LAYERS,
        pretrained_embeddings=embedding_weights,
        freeze_embeddings=FREEZE_EMBEDDINGS
    ).to(device)
else:
    model = MLPClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_HIDDEN_LAYERS).to(device)

# -----------------------------
# Entraînement du modèle
print(f"Hyperparamètres : \n{hyperparams}")
train(model, dataset, batch_size=64, num_epochs=3, learning_rate=0.001, train_ratio=0.8, plot_window_size=1000)

# Évaluation du modèle
accuracy, report, conf_matrix = evaluate_model(model, dataset, device, batch_size=32)
print("Accuracy globale : {:.2f}%\n".format(accuracy * 100))
print("\nRapport de classification :\n")
print(report)
itoc = {index: label for label, index in dataset.ctoi.items()}
classes = [itoc[i] for i in range(len(itoc))]
plot_confusion_matrix(conf_matrix, classes)
analyze_errors(model, dataset, device, batch_size=32, num_examples=5)

# Sauvegarde du modèle entraîné
checkpoint = {
    'model_state_dict': model.state_dict(),
    'hyperparameters': hyperparams,
    'dataset_filters': filters
}
torch.save(checkpoint, "trained_models/test_w2v.pth")
