
import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, n_features, hidden_dim=64):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        # Two stacked LSTM layers for deeper temporal feature extraction
        self.lstm1 = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        
    def forward(self, x):
        x, (_, _) = self.lstm1(x)
        x, (hidden_n, _) = self.lstm2(x)
        # Return the final hidden state (the compressed representation)
        return hidden_n.squeeze(0)

class LSTMDecoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(LSTMDecoder, self).__init__()
        self.seq_len = seq_len
        self.lstm1 = nn.LSTM(hidden_dim // 2, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim * 2, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim * 2, n_features)
        
    def forward(self, x):
        # Repeat the compressed state across the time sequence length
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (_, _) = self.lstm1(x)
        x, (_, _) = self.lstm2(x)
        # Reconstruct the original features
        return self.output_layer(x)

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(n_features, hidden_dim)
        self.decoder = LSTMDecoder(seq_len, n_features, hidden_dim)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":
    # Quick test to ensure the tensor shapes match
    batch_size, seq_length, n_features = 32, 50, 4 # e.g., 50 timesteps of 4 features
    dummy_input = torch.rand((batch_size, seq_length, n_features))
    
    model = LSTMAutoencoder(seq_len=seq_length, n_features=n_features)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") 
    # Output should exactly match Input: [32, 50, 4]
