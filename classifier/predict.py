import torch
from models.mlp_classifier import MLPClassifier
from article_dataset import ArticleDataset
from torch.nn.utils.rnn import pad_sequence

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
    model.eval()  # because we are not training it
    return model, hyperparams

def preprocess_text(text, wtoi):
    tokens = text.split()
    indices = [wtoi.get(token, wtoi.get("<UNK>", 0)) for token in tokens]
    return torch.tensor(indices, dtype=torch.long)

def predict(text, model, wtoi, device):
    # Preprocessing: converting the text to a tensor and adding a batch dimension
    input_tensor = preprocess_text(text, wtoi).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()
    return prediction

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = "trained_models/test.pth"
    model, hyperparams = load_checkpoint(checkpoint_path, device)
    
    # Load the dataset to retrieve the wtoi and ctoi mappings from the dataset
    dataset = ArticleDataset("data/sci_papers.csv")
    
    text = "Importance of words in english"
    prediction_idx = predict(text, model, dataset.wtoi, device)
    
    itoc = {index: label for label, index in dataset.ctoi.items()}
    predicted_class = itoc.get(prediction_idx, "Unknown class")
    
    print("Text :", text)
    print("Prediction (index) :", prediction_idx)
    print("Predicted class :", predicted_class)

if __name__ == "__main__":
    main()
