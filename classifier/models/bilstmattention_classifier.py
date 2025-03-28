# models/bilstmattention_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1,
                 dropout=0.2, pretrained_embeddings=None, freeze_embeddings=False):
        """
        Args:
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Embedding dimension.
            hidden_dim (int): LSTM hidden state dimension.
            num_classes (int): Number of output classes.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability.
            pretrained_embeddings (Tensor, optional): Pretrained embedding matrix.
            freeze_embeddings (bool): If True, embeddings are not updated during training.
        """
        super(BiLSTMAttentionClassifier, self).__init__()
        
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
        
        # Output layer: final representation size is 2 * hidden_dim
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        
    def forward(self, padded_sequences):
        """
        Forward pass with attention mechanism.
        Args:
            padded_sequences (Tensor): [batch_size, seq_len]
        Returns:
            logits (Tensor): [batch_size, num_classes]
        """
        # Pass through embedding layer
        emb = self.embedding(padded_sequences)
        lstm_out, (h_n, _) = self.lstm(emb)
        
        # For bidirectional LSTM, get last layer states for both directions
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        # Use the final hidden state as query for attention
        Q = final_hidden.unsqueeze(1)  # [batch_size, 1, 2*hidden_dim]
        K = lstm_out                   # [batch_size, seq_len, 2*hidden_dim]
        V = lstm_out                   # [batch_size, seq_len, 2*hidden_dim]
        
        d_k = K.size(-1)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)  # [batch_size, 1, seq_len]
        attn_weights = F.softmax(attn_scores, dim=-1)                    # [batch_size, 1, seq_len]
        context = torch.bmm(attn_weights, V).squeeze(1)                   # [batch_size, 2*hidden_dim]
        context = self.dropout(context)
        logits = self.fc(context)
        return logits

    def __len__(self):
        return sum(p.numel() for p in self.parameters())
