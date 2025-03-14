import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_hidden_layers=1):
        """
        Args:
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Embedding dimension.
            hidden_dim (int): Hidden layer dimension.
            num_classes (int): Number of output classes.
            num_hidden_layers (int): Number of hidden layers (0 = no hidden layer).
        """
        super(MLPClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # If there are no hidden layers, pass directly from the embedding to the output
        if num_hidden_layers == 0:
            self.output_layer = nn.Linear(embedding_dim, num_classes)
        else:
            self.input_layer = nn.Linear(embedding_dim, hidden_dim)
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)]
            )
            self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, padded_sequences):
        """
        Args:
            padded_sequences : Tensor of shape [batch_size, seq_len]
        Returns:
            logits : Tensor of shape [batch_size, num_classes]
        """
        # Apply embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        emb = self.embedding(padded_sequences)
        
        # Create a mask to ignore padding (here index 0)
        mask = (padded_sequences != 0).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        emb = emb * mask
        
        # Mean pooling: sum over the temporal (i.e word sequence) dimension divided by the number of real tokens
        sum_emb = emb.sum(dim=1)            # [batch_size, embedding_dim]
        lengths = mask.sum(dim=1)           # [batch_size, 1]
        avg_emb = sum_emb / lengths.clamp(min=1)
        
        # Pass through the MLP
        if hasattr(self, 'input_layer'):
            out = F.relu(self.input_layer(avg_emb))
            if hasattr(self, 'hidden_layers'):
                for layer in self.hidden_layers:
                    out = F.relu(layer(out))
            logits = self.output_layer(out)
        else:
            logits = self.output_layer(avg_emb)
        return logits
    
    def __len__(self):
        return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from article_dataset import ArticleDataset
    from models.mlp_classifier import MLPClassifier

    csv_file = "../data/sci_papers.csv"  
    dataset = ArticleDataset(csv_file)

    filters = {
        "min_papers" : 5000, 
        "min_freq": 2,
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
    model = MLPClassifier(vocab_size, embedding_dim, hidden_dim, num_classes, num_hidden_layers)
    
    print(len(model))
    for p in model.parameters():
        print(p.shape)
   