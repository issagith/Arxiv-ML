from article_dataset import ArticleDataset
from models.mlp_classifier import MLPClassifier
from train import train 
from eval import evaluate_model, plot_confusion_matrix, analyze_errors
import torch 

# -----------------------------
csv_file = "data/articles.csv"  

# classification level:
classification_level = "category" # either category or sub_category
print("dataset initialization...")
dataset = ArticleDataset(csv_file, classification_level)

# dataset filters: 
filters = {
    "min_papers" : 5000, # min nbr of papers in a given category
    "min_freq": 2, # min nbr of apparition of each word to be included in the vocabulary
}
dataset.apply_filters(filters)
print("dataset ready !")

# hyperparameters: 
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

# device:
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------
model = MLPClassifier(vocab_size, embedding_dim, hidden_dim, num_classes, num_hidden_layers).to(device)

# full training
print(f"hyperparameters : \n {hyperparams}")
train(model, dataset, batch_size=64, num_epochs=3, learning_rate=0.001, train_ratio=0.8, plot_window_size=1000)

# model evaluation 
accuracy, report, conf_matrix = evaluate_model(model, dataset, device, batch_size=32)
print("Overall Accuracy : {:.2f}%\n".format(accuracy * 100))
print(("\nClassification Report:\n"))
print(report)
itoc = {index: label for label, index in dataset.ctoi.items()}
classes = [itoc[i] for i in range(len(itoc))]
plot_confusion_matrix(conf_matrix, classes)
analyze_errors(model, dataset, device, batch_size=32, num_examples=5)

# load
checkpoint = {
    'model_state_dict': model.state_dict(),
    'hyperparameters': hyperparams,
    'dataset_filters' : filters
}

torch.save(checkpoint, "trained_models/test_main.pth")
