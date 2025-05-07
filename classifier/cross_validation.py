import torch
import logging
import statistics
from sklearn.model_selection import KFold, ShuffleSplit, ParameterGrid, ParameterSampler
from torch.utils.data import Subset, DataLoader
from utils import custom_collate

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, samples = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward(); optimizer.step()
        bs = targets.size(0)
        total_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        samples += bs
    return total_loss / samples, correct / samples * 100


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, samples = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            bs = targets.size(0)
            total_loss += loss.item() * bs
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            samples += bs
    return total_loss / samples, correct / samples * 100


def cross_validate(model_class, dataset, hyperparams_list, splitter,
                   k_folds, num_epochs, learning_rate, batch_size,
                   device, return_history):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    labels = [dataset[i][1].item() if torch.is_tensor(dataset[i][1]) else dataset[i][1] for i in range(len(dataset))]
    data_source = dataset.data

    results = []
    all_history = [] if return_history else None
    vocab_size = len(dataset.word_to_index)
    num_classes = len(dataset.class_to_index)

    for params in hyperparams_list:
        fold_metrics = []
        fold_history = [] if return_history else None
        logging.info(f"Starting CV for params: {params}")

        for fold, (train_idx, val_idx) in enumerate(splitter.split(data_source, labels), start=1):
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                                      collate_fn=custom_collate, shuffle=True)
            val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size,
                                    collate_fn=custom_collate, shuffle=False)

            model = model_class(vocab_size=vocab_size, num_classes=num_classes, **params).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            if return_history:
                history = {'train_loss': [], 'train_accuracy': [],
                           'val_loss': [], 'val_accuracy': []}

            for epoch in range(1, num_epochs + 1):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)

                if return_history:
                    history['train_loss'].append(train_loss)
                    history['train_accuracy'].append(train_acc)
                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_acc)
                logging.info(
                    f"Fold {fold}/{k_folds} Epoch {epoch} — "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )

            fold_metrics.append({'val_loss': val_loss, 'val_accuracy': val_acc})
            if return_history:
                fold_history.append(history)

        losses = [m['val_loss'] for m in fold_metrics]
        accs = [m['val_accuracy'] for m in fold_metrics]
        avg_loss = sum(losses) / k_folds
        std_loss = statistics.stdev(losses) if k_folds > 1 else 0.0
        avg_acc = sum(accs) / k_folds
        std_acc = statistics.stdev(accs) if k_folds > 1 else 0.0

        results.append({'params': params,
                        'avg_val_loss': avg_loss,
                        'std_val_loss': std_loss,
                        'avg_val_accuracy': avg_acc,
                        'std_val_accuracy': std_acc})
        if return_history:
            all_history.append(fold_history)
        logging.info(f"Params {params} — Avg Loss: {avg_loss:.4f} ± {std_loss:.4f}, Avg Acc: {avg_acc:.2f}% ± {std_acc:.2f}%")

    return (results, all_history) if return_history else results


if __name__ == "__main__":
    from models.mlp_classifier import MLPClassifier
    from article_dataset import ArticleDataset

    # Dataset setup
    dataset = ArticleDataset("data/articles.csv", classification_level="category")
    dataset.apply_filters({"max_papers": 30000, "min_papers": 5000, "min_freq": 5})

    # Common params\>n    param_grid = {'embedding_dim': [64], 'hidden_dim': [128], 'num_hidden_layers': [2], 'dropout': [0.3, 0.5]}
    param_grid = {'embedding_dim': [64, 128, 256], 'hidden_dim': [128, 256, 512, 1024],
                  'num_hidden_layers': [2, 3, 4], 'dropout': [0.3, 0.5]}
    
    k_folds, num_epochs, lr, bs = 5, 10, 0.001, 128

    device = 'cuda' 

    n_iter = 5
    hp_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))
    splitter = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    results, history = cross_validate(
        MLPClassifier, dataset, hp_list, splitter, k_folds=k_folds, num_epochs=num_epochs,
        learning_rate=lr, batch_size=bs, device=device, return_history=True
    ) 

    # save results  and history to files
    import json
    import os
    with open("experiments/cross_val_results.json", "w") as f:
        json.dump(results, f, indent=4)
    with open("results/cv_history.json", "w") as f:
        json.dump(history, f, indent=4)
    logging.info("Cross-validation results and history saved to 'results' directory.")

