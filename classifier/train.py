import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# Custom collate function for padding
def custom_collate(batch, pad_value=0):
    sequences, labels = zip(*batch)
    # Padding sequences to obtain tensors of the same size
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    labels = torch.tensor(labels)
    return padded_sequences, labels

def train_loop(model, dataloader, criterion, optimizer, device):
    """
    Training loop for a DataLoader.
    Returns the average loss over the DataLoader.
    """
    model.train()
    total_loss = 0.0
    for padded_sequences, labels in dataloader:
        padded_sequences = padded_sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(padded_sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def test_loop(model, dataloader, device):
    """
    Evaluation loop for a DataLoader.
    Returns the accuracy in percentage.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for padded_sequences, labels in dataloader:
            padded_sequences = padded_sequences.to(device)
            labels = labels.to(device)
            outputs = model(padded_sequences)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples * 100
    return accuracy

def train(model, dataset, batch_size=32, num_epochs=10, learning_rate=0.001, train_ratio=0.8, device=None, collate_fn=custom_collate):
    """
    Generic training function that splits the dataset into training and test sets,
    and performs the training and evaluation loops.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Splitting the dataset into training and test sets
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        train_loss = train_loop(model, train_dataloader, criterion, optimizer, device)
        test_accuracy = test_loop(model, test_dataloader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

def main():
    from article_dataset import ArticleDataset
    from models.mlp_classifier import MLPClassifier

    csv_file = "data/articles.csv"  
    dataset = ArticleDataset(csv_file)

    # Model parameters
    vocab_size = len(dataset.wtoi)
    embedding_dim = 128
    hidden_dim = 256
    num_classes = len(dataset.ctoi)
    num_hidden_layers = 1

    model = MLPClassifier(vocab_size, embedding_dim, hidden_dim, num_classes, num_hidden_layers)

    # Model training
    train(model, dataset, batch_size=32, num_epochs=1, learning_rate=0.01, train_ratio=0.8)

if __name__ == "__main__":
    main()
