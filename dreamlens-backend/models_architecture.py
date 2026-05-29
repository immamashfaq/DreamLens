import torch
import torch.nn as nn
import math

# --- PREVIOUS PHASE 1 MODEL ---
# This was my initial attempt using a CNN-LSTM combo
class DreamLens_LSTM(nn.Module):
    def __init__(self, num_classes, in_chans):
        super(DreamLens_LSTM, self).__init__()
        
        # Part 1: The CNN (Feature Extraction)
        # Using a few conv layers to pull out patterns from the raw EEG signal
        self.cnn = nn.Sequential(
            nn.Conv1d(in_chans, 64, kernel_size=64, padding=32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=16, padding=8),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Flatten()
        )
        
        # Part 2: The LSTM (Temporal Logic)
        # The LSTM handles the sequence timing; input size comes from the flattened CNN output
        self.lstm = nn.LSTM(input_size=23936, hidden_size=128, batch_first=True)
        
        # Part 3: Final Classifier
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Reshaping the data so the CNN can process the time steps
        batch_size, seq_len, C, T = x.size()
        c_in = x.view(batch_size * seq_len, C, T)
        features = self.cnn(c_in)
        
        # Feeding the features into the LSTM
        r_in = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(r_in)
        
        # Just taking the last output from the LSTM for the final guess
        out = self.fc(lstm_out[:, -1, :])
        return out

# --- PHASE 2: NOVEL ARCHITECTURE ---
# Switching to a Transformer for better efficiency and long-range dependencies

class PositionalEncoding(nn.Module):
    """Adding some sense of order since Transformers don't know sequence timing by default."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Standard sine/cosine math to encode positions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Just adding the position values to the input features
        x = x + self.pe[:, :x.size(1), :]
        return x

class DreamLens_Transformer(nn.Module):
    def __init__(self, num_classes=5, in_chans=1, d_model=128, nhead=4, num_layers=2):
        super(DreamLens_Transformer, self).__init__()
        
        # 1. Lightweight Feature Extractor
        # Using strided convolutions here to shrink the data size before the Transformer
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_chans, 64, kernel_size=64, stride=8, padding=32),
            nn.BatchNorm1d(64), # Keep things stable
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=32, stride=4, padding=16),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        # 3. Lightweight Transformer Encoder
        # Keeping this small (2 layers, 4 heads) so it runs faster on lower-end hardware
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256, 
            dropout=0.2, # Dropout to prevent overfitting
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. Classifier
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # First, pull out the signal features
        features = self.feature_extractor(x) 
        
        # Need to flip the dimensions so it matches what the Transformer expects
        features = features.permute(0, 2, 1) 
        features = self.pos_encoder(features)
        
        # Pass through the attention layers
        transformer_out = self.transformer_encoder(features)
        
        # Global average pooling to get one vector for the whole sequence
        pooled_out = torch.mean(transformer_out, dim=1)
        
        # Map to the 5 sleep stages
        out = self.fc(pooled_out)
        return out