# models/bilstm_classifier.py
import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1,
                 dropout=0.2, pretrained_embeddings=None, freeze_embeddings=False):
        super(BiLSTMClassifier, self).__init__()
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, 
                                                          freeze=freeze_embeddings,
                                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.dropout = nn.Dropout(dropout)
        
        # Bidirectional LSTM
        # Input: [batch_size, seq_len, embedding_dim]
        # Output: lstm_out [batch_size, seq_len, 2*hidden_dim]
        #         h_n [2*num_layers, batch_size, hidden_dim]
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        
        # Linear layer: input [batch_size, 2*hidden_dim] -> output [batch_size, num_classes]
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        
    def forward(self, padded_sequences):
        """
        Forward pass.
        Args:
            padded_sequences (Tensor): [batch_size, seq_len]
        Returns:
            logits (Tensor): [batch_size, num_classes]
        """
        emb = self.embedding(padded_sequences) # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        
        # lstm_out: [batch_size, seq_len, 2*hidden_dim]
        # h_n: [2*num_layers, batch_size, hidden_dim]
        lstm_out, (h_n, _) = self.lstm(emb)
        
        if self.lstm.bidirectional:
            forward_hidden = h_n[-2] # [batch_size, hidden_dim]
            backward_hidden = h_n[-1] # [batch_size, hidden_dim]
            final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1) # [batch_size, 2*hidden_dim]
        else:
            final_hidden = h_n[-1]
            
        out = self.dropout(final_hidden)
        logits = self.fc(out)
        return logits

    def __len__(self):
        return sum(p.numel() for p in self.parameters())
