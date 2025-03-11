import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import numpy as np  
import os
import time 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training_logs/training_log.txt", mode="w")],
    encoding='utf-8'
)

# Custom collate function for padding
def custom_collate(batch, pad_value=0):
    sequences, labels = zip(*batch)
    # Padding sequences to obtain tensors of uniform size
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    labels = torch.tensor(labels)
    return padded_sequences, labels


def train_loop(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    iteration_losses = []
    size = len(dataloader.dataset)
    for batch, (padded_sequences, labels) in enumerate(dataloader):
        padded_sequences = padded_sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(padded_sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()
        total_loss += loss_value
        iteration_losses.append(loss_value)
        
        if batch % 100 == 0:
            current = batch * dataloader.batch_size + len(padded_sequences)
            logging.info(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
    avg_loss = total_loss / len(dataloader)
    return avg_loss, iteration_losses


def test_loop(model, dataloader, criterion, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    with torch.no_grad():
        for padded_sequences, labels in dataloader:
            padded_sequences = padded_sequences.to(device)
            labels = labels.to(device)
            outputs = model(padded_sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * padded_sequences.size(0)  # Accumulates the loss weighted by the batch size to get the right avg
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


def moving_average(data, window_size=1000):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float)

    cumsum = data.cumsum(0)
    cumsum = torch.cat((torch.tensor([0.], device=data.device, dtype=data.dtype), cumsum))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return moving_avg


def plot_loss_curve(iteration_losses, window_size=1000):
    smoothed_losses = moving_average(iteration_losses, window_size)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(iteration_losses)), iteration_losses, alpha=0.3, label="Loss per iteration")
    plt.plot(range(window_size - 1, len(iteration_losses)), smoothed_losses, color='red', label="Moving average")
    plt.title("Loss Curve over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"training_plots/loss_curve_{timestamp}.png"
    plt.savefig(filename)
    plt.show()
    logging.info(f"The loss curve has been saved as '{filename}'.")


def train(model, dataset, batch_size=32, num_epochs=10, learning_rate=0.001, train_ratio=0.8, device=None, collate_fn=custom_collate, plot_window_size=1000):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Splitting the dataset into training and testing sets
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Global list to record the loss at every iteration
    global_iteration_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss, iteration_losses = train_loop(model, train_dataloader, criterion, optimizer, device)
        global_iteration_losses.extend(iteration_losses)
        test_loss, test_accuracy = test_loop(model, test_dataloader, criterion, device)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss (epoch average): {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    plot_loss_curve(global_iteration_losses, window_size=plot_window_size)

def main():
    from article_dataset import ArticleDataset
    from models.mlp_classifier import MLPClassifier

    csv_file = "data/articles.csv"  
    dataset = ArticleDataset(csv_file)

    filters = {
        "min_papers" : 5000, 
        "min_freq": 5,
    }
    dataset.apply_filters(filters)

    vocab_size = len(dataset.wtoi)
    embedding_dim = 64
    hidden_dim = 512
    num_classes = len(dataset.ctoi)
    num_hidden_layers = 1

    hyperparams = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'num_hidden_layers': num_hidden_layers
    }

    logging.info(f"starting training with the following hyperparams :")
    for p, v in hyperparams.items():
        logging.info(f"{p} : {v}")
    logging.info('\n')
    model = MLPClassifier(vocab_size, embedding_dim, hidden_dim, num_classes, num_hidden_layers)

    train(model, dataset, batch_size=64, num_epochs=1, learning_rate=0.001, train_ratio=0.8, plot_window_size=1000)
    
    # Saving the model and hyperparameters
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'hyperparameters': hyperparams
    }
    #torch.save(checkpoint, "trained_models/small_dataset_model.pth")"
    

if __name__ == "__main__":
    main()
