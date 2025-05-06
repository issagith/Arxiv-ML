# eval.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from models.mlp_classifier import MLPClassifier
from article_dataset import ArticleDataset
from utils import custom_collate, load_checkpoint, preprocess_text
import os

def evaluate_model(model, dataset, device, batch_size=32):
    """
    Evaluates the model on the given dataset.
    Returns accuracy, classification report, and confusion matrix.
    """
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)

     # If dataset is a Subset, get the original dataset attributes.
    if hasattr(dataset, "index_to_class"):
        index_to_class = dataset.index_to_class
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "index_to_class"):
        index_to_class = dataset.dataset.index_to_class
    else:
        raise AttributeError("The provided dataset does not have an 'index_to_class' attribute.")
    labels = list(range(len(index_to_class)))
    target_names = [index_to_class[i] for i in labels]
    report = classification_report(all_labels, all_preds, labels=labels, target_names=target_names)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, conf_matrix

def plot_confusion_matrix(conf_matrix, classes, output_path=None):
    """
    Plots a confusion matrix heatmap.
    If output_path is specified, saves the plot to that file.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predictions")
    plt.ylabel("True Classes")
    plt.title("Confusion Matrix")
    if output_path:
        plt.savefig(output_path)
        print(f"[INFO] Confusion matrix plot saved as '{output_path}'.")
    plt.show()

def analyze_errors(model, dataset, device, batch_size=32, num_examples=5):
    """
    Displays examples of misclassified texts.
    """
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    misclassified = []
    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(device)
            outputs = model(texts)
            preds = outputs.argmax(dim=1)
            for i in range(len(labels)):
                if preds[i].item() != labels[i].item():
                    words = [dataset.index_to_word.get(idx, "<unk>") for idx in texts[i].cpu().numpy() if idx != 0]
                    misclassified.append((" ".join(words), labels[i].item(), preds[i].item()))
            if len(misclassified) >= num_examples:
                break
    if misclassified:
        print("\nExamples of misclassified texts:")
        for text, true_idx, pred_idx in misclassified[:num_examples]:
            print("Text       :", text)
            print("True Label :", dataset.index_to_class.get(true_idx, "Unknown"))
            print("Predicted  :", dataset.index_to_class.get(pred_idx, "Unknown"))
            print("-----")
    else:
        print("\nNo misclassified examples found.")

def plot_roc_curve(model, dataset, device, batch_size=32):
    """
    Plots the ROC curve for binary classification.
    """
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(device)
            outputs = model(texts)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

