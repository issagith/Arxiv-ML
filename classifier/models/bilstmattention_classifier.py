import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1,
                 dropout=0.2, pretrained_embeddings=None, freeze_embeddings=False):
        """
        Args:
            vocab_size (int): Taille du vocabulaire.
            embedding_dim (int): Dimension des embeddings.
            hidden_dim (int): Dimension des états cachés du LSTM.
            num_classes (int): Nombre de classes de sortie.
            num_layers (int): Nombre de couches LSTM.
            dropout (float): Probabilité de dropout.
            pretrained_embeddings (Tensor, optionnel): Matrice d'embeddings pré-entraînés.
            freeze_embeddings (bool): Si True, les embeddings ne seront pas mis à jour pendant l'entraînement.
        """
        super(BiLSTMAttentionClassifier, self).__init__()
        
        # Couche d'embedding avec gestion du padding (index 0)
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, 
                                                          freeze=freeze_embeddings,
                                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.dropout = nn.Dropout(dropout)
        
        # LSTM bidirectionnel
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        
        # Couche de sortie : La dimension finale est 2 * hidden_dim
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        
    def forward(self, padded_sequences):
        """
        Args:
            padded_sequences: Tensor de forme [batch_size, seq_len]
        Returns:
            logits: Tensor de forme [batch_size, num_classes]
        """
        # Passage par l'embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        emb = self.embedding(padded_sequences)
        
        # Passage par le LSTM bidirectionnel
        # lstm_out : [batch_size, seq_len, 2*hidden_dim]
        # (h_n, c_n) sont respectivement les derniers états cachés et les états de cellule
        lstm_out, (h_n, c_n) = self.lstm(emb)
        
        # Pour un LSTM bidirectionnel, h_n a la forme [num_layers * 2, batch_size, hidden_dim].
        # Les deux dernières lignes correspondent aux états de la dernière couche :
        # - h_n[-2] pour la direction forward
        # - h_n[-1] pour la direction backward
        forward_hidden = h_n[-2]  # [batch_size, hidden_dim]
        backward_hidden = h_n[-1]  # [batch_size, hidden_dim]
        final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)  # [batch_size, 2*hidden_dim]
        
        # Utilisation du dernier état caché comme requête (Query)
        # On reforme Q pour qu'il soit de forme [batch_size, 1, 2*hidden_dim]
        Q = final_hidden.unsqueeze(1)
        # Les sorties du LSTM servent de clés (K) et valeurs (V)
        K = lstm_out  # [batch_size, seq_len, 2*hidden_dim]
        V = lstm_out  # [batch_size, seq_len, 2*hidden_dim]
        
        d_k = K.size(-1)  # d_k = 2 * hidden_dim
        # Calcul des scores d'attention (scaled dot-product)
        # Score = (Q • K^T) / sqrt(d_k)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)  # [batch_size, 1, seq_len]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, 1, seq_len]
        
        # Calcul du vecteur de contexte comme somme pondérée des valeurs
        context = torch.bmm(attn_weights, V)  # [batch_size, 1, 2*hidden_dim]
        context = context.squeeze(1)  # [batch_size, 2*hidden_dim]
        
        # Optionnel : appliquer le dropout sur le vecteur de contexte
        context = self.dropout(context)
        
        # Passage par la couche de sortie pour obtenir les logits de classification
        logits = self.fc(context)  # [batch_size, num_classes]
        return logits

    def __len__(self):
        return sum(p.numel() for p in self.parameters())
