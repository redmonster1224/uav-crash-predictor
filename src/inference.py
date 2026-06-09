
import torch
import numpy as np
import os
import time
from model import LSTMAutoencoder

class UAVSafetyEngine:
    def __init__(self, model_path, seq_len=50, n_features=4, hidden_dim=64):
        self.seq_len = seq_len
        self.n_features = n_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize the architecture
        self.model = LSTMAutoencoder(seq_len=seq_len, n_features=n_features, hidden_dim=hidden_dim)
        
        # 2. Load the trained weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_for_mapping=self.device))
            print(f"Successfully loaded SOTA safety weights from {model_path}")
        else:
            print(f"Warning: Weights not found at {model_path}. Running with uninitialized weights for testing.")
            
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode (turns off dropout/batchnorm updates)
        
        # 3. Create a rolling FIFO queue/buffer to hold live telemetry stream
        self.buffer = []
        
        # Set an empirical anomaly threshold (learned during training evaluation)
        self.threshold = 0.05 

    def update_stream(self, raw_telemetry_vector):
        """
        Ingests a single timestep of telemetry from the live UAV stream.
        Vector format: [Pitch, Roll, Motor_RPM, Battery_Voltage]
        """
        self.buffer.append(raw_telemetry_vector)
        
        # Maintain our lookback window size
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)
            
        # We can only perform inference once the buffer is full
        if len(self.buffer) == self.seq_len:
            return self.evaluate_anomaly()
        
        return 0.0, "BUFFERING"

    def evaluate_anomaly(self):
        """
        Passes the current rolling window through the Autoencoder 
        and calculates the real-time Reconstruction Error (MSE).
        """
        # Convert buffer to tensor and add batch dimension: (1, seq_len, n_features)
        sequence = np.array(self.buffer)
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad(): # Disable gradient calculation for fast execution
            reconstructed = self.model(sequence_tensor)
            
            # Calculate MSE between the original sequence and the reconstruction
            loss = torch.mean((sequence_tensor - reconstructed) ** 2).item()
            
        # Determine safety state
        status = "NOMINAL"
        if loss > self.threshold:
            status = "CRITICAL: ANOMALOUS SYSTEM BEHAVIOR DETECTED"
            
        return loss, status

if __name__ == "__main__":
    # Test the real-time engine with a mock stream
    engine = UAVSafetyEngine(model_path="../models/lstm_autoencoder_v1.pth")
    
    print("\nSimulating real-time UAV flight stream...")
    for step in range(70):
        if step < 60:
            # Simulate stable flight data
            live_vector = np.random.normal(0, 0.05, 4).tolist()
        else:
            # Simulate a sudden mechanical failure or severe aerodynamic stall
            print("\n!!! INTRODUCING SIMULATED HARDWARE FAULT !!!")
            live_vector = [1.5, -2.0, 8.5, -3.0] # Massive structural/electrical deviation
            
        score, state = engine.update_stream(live_vector)
        print(f"Timestep {step:02d} | Anomaly Score (MSE): {score:.6f} | Status: {state}")
        time.sleep(0.05) # Simulate 20Hz stream
