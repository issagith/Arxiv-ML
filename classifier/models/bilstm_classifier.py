import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1,
                 dropout=0.2, pretrained_embeddings=None, freeze_embeddings=False):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of embeddings.
            hidden_dim (int): Dimension of LSTM hidden states.
            num_classes (int): Number of output classes.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability.
            pretrained_embeddings (Tensor, optional): Pre-trained embeddings matrix.
            freeze_embeddings (bool): If True, embeddings won't be updated during training.
        """
        super(BiLSTMClassifier, self).__init__()
        
        # Embedding layer with padding handling (index 0)
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, 
                                                        freeze=freeze_embeddings,
                                                        padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.dropout = nn.Dropout(dropout)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)
        
        # Output layer: final representation has size 2 * hidden_dim
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        
    def forward(self, padded_sequences):
        """
        Args:
            padded_sequences: Tensor of shape [batch_size, seq_len]
        Returns:
            logits: Tensor of shape [batch_size, num_classes]
        """
        # Pass through embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        emb = self.embedding(padded_sequences)
        
        # Pass through bidirectional LSTM
        # lstm_out has shape [batch_size, seq_len, 2*hidden_dim]
        # (h_n, c_n) are hidden states and cell states respectively
        lstm_out, (h_n, c_n) = self.lstm(emb)
        # h_n has shape [num_layers * num_directions, batch_size, hidden_dim]
        # For bidirectional LSTM, last two rows correspond to last layer states:
        # h_n[-2] for forward direction and h_n[-1] for backward direction
        if self.lstm.bidirectional:
            forward_hidden = h_n[-2]   # shape: [batch_size, hidden_dim]
            backward_hidden = h_n[-1]   # shape: [batch_size, hidden_dim]
            final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)  # [batch_size, 2*hidden_dim]
        else:
            final_hidden = h_n[-1]
        
        out = self.dropout(final_hidden)
        logits = self.fc(out)
        return logits

    def __len__(self):
        return sum(p.numel() for p in self.parameters())