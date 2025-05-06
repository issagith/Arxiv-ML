# main.py
import os
import torch
from article_dataset import ArticleDataset
from models.mlp_classifier import MLPClassifier
from models.bilstm_classifier import BiLSTMClassifier
from models.bilstmattention_classifier import BiLSTMAttentionClassifier
from train import train
from eval import evaluate_model, plot_confusion_matrix
from gensim.models import KeyedVectors
from utils import build_embedding_matrix
import logging
from torch.utils.data import DataLoader
from utils import custom_collate
import article_dataset

# Ask the user for an experiment name
experiment_name = input("Enter experiment name: ").strip()
if not experiment_name:
    experiment_name = "default_experiment"

# Create the experiment directory inside trained_models
experiment_dir = os.path.join("experiments", experiment_name)
os.makedirs(experiment_dir, exist_ok=True)

# Create subdirectories for logs and plots inside the experiment folder
logs_dir = os.path.join(experiment_dir, "logs")
plots_dir = os.path.join(experiment_dir, "plots")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Configure logging to save logs in the experiment's logs folder
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(logs_dir, "training_log.txt"), mode="w", encoding="utf-8")
    ]
)

# -----------------------------
# Constants and general parameters
CSV_FILE = "data/articles.csv"
CLASSIFICATION_LEVEL = "category"  # "category" or "sub_category"
MODEL = "mlp"
MODELS = {
    "bilstm": BiLSTMClassifier,
    "bilstm_attention": BiLSTMAttentionClassifier,
    "mlp": MLPClassifier
}

# Model hyperparameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_HIDDEN_LAYERS = 3
batch_size = 128
IS_CUSTOM_EMB = True
FREEZE_EMBEDDINGS = False
DROPOUT = 0.3

# -----------------------------
# Dataset initialization
print("[INFO] Initializing dataset...")
dataset = ArticleDataset(CSV_FILE, classification_level=CLASSIFICATION_LEVEL)

# Applying filters to the dataset
filters = {
    "max_papers": 5000,  
    "min_freq": 2,       
}
dataset.apply_filters(filters)
print("[INFO] Dataset ready!")

# Shuffle the dataset randomly before splitting into train and test sets
from torch.utils.data import random_split

train_ratio = 0.8  # Must be the same ratio used in training
train_size = int(len(dataset) * train_ratio)
test_size = len(dataset) - train_size

# Perform a random split of the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print("[DEBUG] article_dataset module loaded from:", article_dataset.__file__)
print("[DEBUG] classification_level:", dataset.classification_level)
print("[DEBUG] classes mapping:", dataset.index_to_class)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate)

# Determining dimensions
VOCAB_SIZE = len(dataset.word_to_index)
NUM_CLASSES = len(dataset.class_to_index)

hyperparams = {
    'vocab_size': VOCAB_SIZE,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': HIDDEN_DIM,
    'num_classes': NUM_CLASSES,
    'num_hidden_layers': NUM_HIDDEN_LAYERS,
    'dropout': DROPOUT,
    'freeze_embeddings': FREEZE_EMBEDDINGS
}

# Choosing the device (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Loading Word2Vec embeddings from the local file
if not IS_CUSTOM_EMB:
    w2v_path = os.path.join("data", "word2vec-google-news-300.bin")
    if os.path.exists(w2v_path):
        print("[INFO] Loading Word2Vec model from local file...")
        w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    else:
        raise FileNotFoundError(f"The file {w2v_path} was not found. Please ensure it is present.")

    # Build the embedding matrix using the helper function from utils.py
    embedding_weights, missing_words = build_embedding_matrix(dataset.word_to_index, w2v_model, EMBEDDING_DIM)
    print(f"[INFO] {missing_words}/{len(dataset.word_to_index)} words are not present in Word2Vec.")

    # Create the model with pretrained embeddings
    model = MODELS[MODEL](
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
    model = MODELS[MODEL](VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_HIDDEN_LAYERS, DROPOUT).to(device)

# -----------------------------
# Training the model
print(f"[INFO] Hyperparameters: {hyperparams}")
print(f"[INFO] model size: {len(model)} parameters")
# Pass the experiment directory so that logs/plots are saved in the proper subfolders
train(model, train_dataloader, test_dataloader, num_epochs=5, learning_rate=0.001, plot_window_size=1000, output_dir=experiment_dir)

# Save the model
checkpoint = {
    'model_state_dict': model.state_dict(),
    'hyperparameters': hyperparams,
    'dataset_filters': filters
}
model_save_path = os.path.join(experiment_dir, f"{experiment_name}.pth")
torch.save(checkpoint, model_save_path)
print(f"[INFO] Model saved at {model_save_path}")

# -----------------------------

# Evaluation on the training set
print("[INFO] Evaluating on training set...")
train_acc, train_report, train_conf_matrix = evaluate_model(model, train_dataset, device, batch_size=32)
train_eval_path = os.path.join(experiment_dir, "evaluation_train.txt")
with open(train_eval_path, "w", encoding="utf-8") as f:
    f.write("Overall Accuracy : {:.2f}%\n".format(train_acc * 100))
    f.write("\nClassification Report:\n")
    f.write(train_report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(train_conf_matrix))
print(f"[INFO] Training set evaluation saved to {train_eval_path}")

# Evaluation on the test set
print("[INFO] Evaluating on test set...")
test_acc, test_report, test_conf_matrix = evaluate_model(model, test_dataset, device, batch_size=32)
test_eval_path = os.path.join(experiment_dir, "evaluation_test.txt")
with open(test_eval_path, "w", encoding="utf-8") as f:
    f.write("Overall Accuracy : {:.2f}%\n".format(test_acc * 100))
    f.write("\nClassification Report:\n")
    f.write(test_report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(test_conf_matrix))
print(f"[INFO] Test set evaluation saved to {test_eval_path}")

# Save confusion matrices in the plots folder
classes = [dataset.index_to_class[i] for i in range(len(dataset.index_to_class))]
train_cm_path = os.path.join(plots_dir, "confusion_matrix_train.png")
plot_confusion_matrix(train_conf_matrix, classes, output_path=train_cm_path)
test_cm_path = os.path.join(plots_dir, "confusion_matrix_test.png")
plot_confusion_matrix(test_conf_matrix, classes, output_path=test_cm_path)