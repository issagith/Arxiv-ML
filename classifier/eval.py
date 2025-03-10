import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

from models.mlp_classifier import MLPClassifier
from article_dataset import ArticleDataset

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hyperparams = checkpoint['hyperparameters']
    
    model = MLPClassifier(
        hyperparams['vocab_size'],
        hyperparams['embedding_dim'],
        hyperparams['hidden_dim'],
        hyperparams['num_classes'],
        hyperparams['num_hidden_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # evaluation mode
    return model, hyperparams

def preprocess_text(text, wtoi):
    tokens = text.split()
    indices = [wtoi.get(token, wtoi.get("<UNK>", 0)) for token in tokens]
    return torch.tensor(indices, dtype=torch.long)

def custom_collate(batch, pad_value=0):
    sequences, labels = zip(*batch)
    # Pad sequences to obtain uniform tensors
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    labels = torch.tensor(labels)
    return padded_sequences, labels

def evaluate_model(model, dataset, device, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=custom_collate)
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
    itoc = {index: label for label, index in dataset.ctoi.items()}
    
    classes = [itoc[i] for i in range(len(itoc))]
    report = classification_report(all_labels, all_preds, target_names=classes)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, conf_matrix

def plot_confusion_matrix(conf_matrix, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predictions")
    plt.ylabel("True Classes")
    plt.title("Confusion Matrix")
    plt.show()

def analyze_errors(model, dataset, device, batch_size=32, num_examples=5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=custom_collate)
    
    misclassified = []
    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(device)
            outputs = model(texts)
            preds = outputs.argmax(dim=1)
            for i in range(len(labels)):
                if preds[i].item() != labels[i].item():
                    # Reconstruct the text from indices (ignoring padding 0)
                    itow = {idx: word for word, idx in dataset.wtoi.items()}
                    token_ids = texts[i].cpu().numpy()
                    words = [itow.get(idx, "<UNK>") for idx in token_ids if idx != 0]
                    misclassified.append((" ".join(words), labels[i].item(), preds[i].item()))
            if len(misclassified) >= num_examples:
                break
    
    if misclassified:
        print("\nExamples of misclassified texts:")
        itoc = {index: label for label, index in dataset.ctoi.items()}
        for text, true_idx, pred_idx in misclassified[:num_examples]:
            print("Text       :", text)
            print("True Label :", itoc.get(true_idx, "Unknown"))
            print("Predicted  :", itoc.get(pred_idx, "Unknown"))
            print("-----")
    else:
        print("\nNo misclassified examples found.")

def plot_roc_curve(model, dataset, device, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda batch: custom_collate(batch))
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(device)
            outputs = model(texts)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of the positive class
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "trained_models/small_dataset_model.pth"
    
    model, hyperparams = load_checkpoint(checkpoint_path, device)
    dataset = ArticleDataset("data/sci_papers.csv", min_freq=2)
    
    accuracy, report, conf_matrix = evaluate_model(model, dataset, device, batch_size=32)
    with open("eval_results/results.txt", "w", encoding="utf-8") as f:
        f.write("Overall Accuracy : {:.2f}%\n".format(accuracy * 100))
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
    
    itoc = {index: label for label, index in dataset.ctoi.items()}
    classes = [itoc[i] for i in range(len(itoc))]
    plot_confusion_matrix(conf_matrix, classes)
   
    analyze_errors(model, dataset, device, batch_size=32, num_examples=5)
    
    if hyperparams['num_classes'] == 2:
        plot_roc_curve(model, dataset, device, batch_size=32)

if __name__ == "__main__":
    main()
