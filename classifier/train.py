# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import os
import time
import logging
from utils import custom_collate, moving_average, plot_loss_curve

def train_loop(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    iteration_losses = []
    dataset_size = len(dataloader.dataset)
    for batch_idx, (padded_sequences, labels) in enumerate(dataloader):
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
        
        if batch_idx % 100 == 0:
            current = batch_idx * dataloader.batch_size + len(padded_sequences)
            logging.info(f"loss: {loss_value:>7f}  [{current:>5d}/{dataset_size:>5d}]")
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
            total_loss += loss.item() * padded_sequences.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy

def train(model, dataset, batch_size=32, num_epochs=10, learning_rate=0.001,
          train_ratio=0.8, device=None, plot_window_size=1000, output_dir="."):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create directories for logs and plots inside the output directory
    logs_dir = os.path.join(output_dir, "logs")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Split dataset into training and testing subsets
    train_size = int(len(dataset) * train_ratio)
    train_dataset = Subset(dataset, range(0, train_size))
    test_dataset = Subset(dataset, range(train_size, len(dataset)))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    global_iteration_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss, iteration_losses = train_loop(model, train_dataloader, criterion, optimizer, device)
        global_iteration_losses.extend(iteration_losses)
        test_loss, test_accuracy = test_loop(model, test_dataloader, criterion, device)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss (epoch avg): {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Save the loss curve in the plots directory
    plot_loss_curve(global_iteration_losses, window_size=plot_window_size, plots_dir=plots_dir)
