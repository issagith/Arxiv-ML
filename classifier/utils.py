# utils.py
import torch
from torch.nn.utils.rnn import pad_sequence
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def custom_collate(batch, pad_value=0):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    labels = torch.tensor(labels)
    return padded_sequences, labels

def load_checkpoint(checkpoint_path, device, model_class):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hyperparams = checkpoint['hyperparameters']
    hyperparams.pop('is_custom_emb', None)
    model = model_class(**hyperparams)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, hyperparams

def preprocess_text(text, word_to_index):
    tokens = text.split()
    indices = [word_to_index.get(token, word_to_index.get("<unk>", 0)) for token in tokens]
    return torch.tensor(indices, dtype=torch.long)

def moving_average(data, window_size=1000):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float)
    cumsum = data.cumsum(0)
    cumsum = torch.cat((torch.tensor([0.], device=data.device, dtype=data.dtype), cumsum))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return moving_avg

def plot_loss_curve(iteration_losses, window_size=1000, plots_dir="."):
    smoothed_losses = moving_average(iteration_losses, window_size)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(iteration_losses)), iteration_losses, alpha=0.3, label="Loss per iteration")
    plt.plot(range(window_size - 1, len(iteration_losses)), smoothed_losses, color='red', label="Moving average")
    plt.title("Loss Curve over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Ensure the plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(plots_dir, f"loss_curve_{timestamp}.png")
    plt.savefig(filename)
    plt.show()
    print(f"[INFO] The loss curve has been saved as '{filename}'.")

def build_embedding_matrix(word_to_index, w2v_model, embedding_dim):
    vocab_size = len(word_to_index)
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))
    missing_words = 0
    for word, idx in word_to_index.items():
        if word in w2v_model:
            embedding_matrix[idx] = torch.tensor(w2v_model[word])[:embedding_dim]
        else:
            missing_words += 1
            embedding_matrix[idx] = torch.tensor(np.random.randn(embedding_dim), dtype=torch.float)
    return embedding_matrix, missing_words
