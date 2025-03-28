import os
import torch
import numpy as np
from article_dataset import ArticleDataset
from models.mlp_classifier import MLPClassifier
from train import train
from eval import evaluate_model, plot_confusion_matrix, analyze_errors
from gensim.models import KeyedVectors

# -----------------------------
# Constants and general parameters
CSV_FILE = "data/articles.csv"
CLASSIFICATION_LEVEL = "category"  # "category" or "sub_category"

# Model hyperparameters
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_HIDDEN_LAYERS = 1
IS_CUSTOM_EMB = False
FREEZE_EMBEDDINGS = False
DROPOUT = 0.2

# -----------------------------
# Dataset initialization
print("Initializing dataset...")
dataset = ArticleDataset(CSV_FILE, CLASSIFICATION_LEVEL)

# Applying filters to the dataset
filters = {
    "min_papers": 5000,  # minimum number of documents per category
    "min_freq": 2,       # minimum frequency of a word to be included in the vocabulary
}
dataset.apply_filters(filters)
print("Dataset ready!")

# Determining dimensions
VOCAB_SIZE = len(dataset.wtoi)
NUM_CLASSES = len(dataset.ctoi)

hyperparams = {
    'vocab_size': VOCAB_SIZE,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': HIDDEN_DIM,
    'num_classes': NUM_CLASSES,
    'num_hidden_layers': NUM_HIDDEN_LAYERS,
    'dropout': DROPOUT,
    'is_custom_emb': IS_CUSTOM_EMB,
    'freeze_embeddings': FREEZE_EMBEDDINGS
}

# Choosing the device (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Loading Word2Vec embeddings from the local file
if not IS_CUSTOM_EMB:
    w2v_path = os.path.join("data", "word2vec-google-news-300.bin")
    if os.path.exists(w2v_path):
        print("Loading Word2Vec model from local file...")
        w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    else:
        raise FileNotFoundError(f"The file {w2v_path} was not found. Please ensure it is present.")

    # Building the embedding matrix (dimensionality reduction from 300 to EMBEDDING_DIM)
    embedding_weights = torch.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    missing_words = 0
    for word, idx in dataset.wtoi.items():
        if word in w2v_model:
            embedding_weights[idx] = torch.tensor(w2v_model[word])[:EMBEDDING_DIM]
        else:
            missing_words += 1
            # Random initialization for missing words
            embedding_weights[idx] = torch.tensor(np.random.randn(EMBEDDING_DIM), dtype=torch.float)
    print(f"{missing_words}/{len(dataset.wtoi)} words are not present in Word2Vec.")

    # Creating the model with pre-trained embeddings
    model = MLPClassifier(
        VOCAB_SIZE,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_CLASSES,
        NUM_HIDDEN_LAYERS,
        dropout=DROPOUT,
        pretrained_embeddings=embedding_weights,
        freeze_embeddings=FREEZE_EMBEDDINGS
    ).to(device)
else:
    model = MLPClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_HIDDEN_LAYERS, DROPOUT).to(device)

# -----------------------------
# Training the model
print(f"Hyperparameters: \n{hyperparams}")
train(model, dataset, batch_size=64, num_epochs=3, learning_rate=0.001, train_ratio=0.8, plot_window_size=1000)

# Evaluating the model
accuracy, report, conf_matrix = evaluate_model(model, dataset, device, batch_size=32)
print("Overall accuracy: {:.2f}%\n".format(accuracy * 100))
print("\nClassification report:\n")
print(report)
itoc = {index: label for label, index in dataset.ctoi.items()}
classes = [itoc[i] for i in range(len(itoc))]
plot_confusion_matrix(conf_matrix, classes)
analyze_errors(model, dataset, device, batch_size=32, num_examples=5)

# Saving the trained model
checkpoint = {
    'model_state_dict': model.state_dict(),
    'hyperparameters': hyperparams,
    'dataset_filters': filters
}
torch.save(checkpoint, "trained_models/test_w2v.pth")
