import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, n_layers=2, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def attention_net(self, lstm_output, final_state):
        """
        Attention mechanism to focus on relevant parts of the input
        """
        # lstm_output shape: (batch_size, seq_len, hidden_dim * 2)
        # final_state shape: (batch_size, hidden_dim * 2)
        
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # attention_output shape: (batch_size, hidden_dim * 2)
        attention_output = torch.sum(attention_weights * lstm_output, dim=1)
        return attention_output

    def forward(self, text):
        # text shape: (batch_size, seq_length)
        
        embedded = self.embedding(text)
        # embedded shape: (batch_size, seq_length, embedding_dim)
        
        lstm_output, (hidden, cell) = self.lstm(embedded)
        # lstm_output shape: (batch_size, seq_length, hidden_dim * 2)
        # hidden shape: (n_layers * 2, batch_size, hidden_dim)
        
        # Get the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # hidden shape: (batch_size, hidden_dim * 2)
        
        # Apply attention
        attention_output = self.attention_net(lstm_output, hidden)
        
        # Pass through final layers
        output = self.fc(attention_output)
        
        return torch.sigmoid(output).squeeze()

    def count_parameters(self):
        """Count the total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)