import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# Import the model you wrote in model.py
from model import LSTMAutoencoder

def create_sequences(data, seq_length):
    """
    Converts 2D telemetry data into 3D sequences for the LSTM.
    Shape changes from (Total_Rows, Features) -> (Samples, Sequence_Length, Features)
    """
    xs = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        xs.append(x)
    return np.array(xs)

def train_model():
    # 1. Configuration & Hyperparameters
    SEQ_LENGTH = 50       # Look at 50 timesteps of history at a time
    HIDDEN_DIM = 64
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    
    # Setup Device (Use GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Load and Prepare Data
    # In practice, you load the CSV saved by data_parser.py
    # df = pd.read_csv('../data/processed/normal_flight_data.csv')
    # For now, we simulate normalized telemetry data (e.g., Pitch, Roll, RPM, Voltage)
    print("Loading normal flight telemetry...")
    n_features = 4
    dummy_normal_data = np.random.normal(0, 0.1, (5000, n_features)) 
    
    # Create temporal sequences
    X_train = create_sequences(dummy_normal_data, SEQ_LENGTH)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor) # Input and Target are the same
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Model, Loss, and Optimizer
    model = LSTMAutoencoder(seq_len=SEQ_LENGTH, n_features=n_features, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Mean Squared Error for reconstruction

    # 4. The Training Loop
    print("Starting training...")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            
            # Calculate how badly the model failed to reconstruct the data
            loss = criterion(reconstructed, batch_x)
            
            # Backward pass (update weights)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{EPOCHS}] | Reconstruction Loss (MSE): {avg_loss:.6f}")

    # 5. Save the trained weights
    model_path = "../models/lstm_autoencoder_v1.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Weights saved to {model_path}")

if __name__ == "__main__":
    train_model()
