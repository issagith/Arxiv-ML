import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_hidden_layers=1,
                 dropout=0.2, pretrained_embeddings=None, freeze_embeddings=False):
        """
        Args:
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Embedding dimension.
            hidden_dim (int): Hidden layer dimension.
            num_classes (int): Number of output classes.
            num_hidden_layers (int): Number of hidden layers (0 = no hidden layers).
            dropout (float): Dropout probability.
            pretrained_embeddings (Tensor, optional): Pretrained embedding matrix of size [vocab_size, embedding_dim].
            freeze_embeddings (bool): If True, pretrained embeddings will not be updated during training.
        """
        super(MLPClassifier, self).__init__()
        
        # Choice between pretrained embeddings or creating a standard embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Define dropout layer to be used after activations
        self.dropout = nn.Dropout(dropout)
        
        # If no hidden layers are used, directly connect embedding to the output
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
        
        # Create a mask to ignore padding (index 0 here)
        mask = (padded_sequences != 0).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        emb = emb * mask
        
        # Mean pooling: sum over the temporal dimension (i.e., the sequence) divided by the number of actual tokens
        sum_emb = emb.sum(dim=1)            # [batch_size, embedding_dim]
        lengths = mask.sum(dim=1)           # [batch_size, 1]
        avg_emb = sum_emb / lengths.clamp(min=1)
        
        # Pass through the MLP with dropout after each activation
        if hasattr(self, 'input_layer'):
            out = F.relu(self.input_layer(avg_emb))
            out = self.dropout(out)  # Apply dropout after the input layer
            if hasattr(self, 'hidden_layers'):
                for layer in self.hidden_layers:
                    out = F.relu(layer(out))
                    out = self.dropout(out)  # Apply dropout after each hidden layer
            logits = self.output_layer(out)
        else:
            logits = self.output_layer(avg_emb)
        return logits
    
    def __len__(self):
        return sum(p.numel() for p in self.parameters())
