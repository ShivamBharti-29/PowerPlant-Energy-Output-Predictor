import torch
import torch.nn as nn
import numpy as np

# This definition must exactly match your notebook's class
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        # In your notebook, you used nn.Sequential named 'self.model'
        self.model = nn.Sequential(
            nn.Linear(4, 6), # Input layer
            nn.ReLU(),
            nn.Linear(6, 6), # Hidden layer
            nn.ReLU(),
            nn.Linear(6, 1)  # Output layer
        )

    def forward(self, x):
        return self.model(x)

def predict_power(data):
    model = ANN()
    try:
        # Load the state dict (weights)
        # Use map_location='cpu' to ensure it works on the Streamlit server
        model.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
        model.eval()
        
        # Data processing
        input_tensor = torch.tensor(data, dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(input_tensor)
        
        return f"{prediction.item():.2f} MW"
    except FileNotFoundError:
        return "Error: best_model.pt not found. Please upload it to your GitHub repo."
    except Exception as e:
        return f"Prediction Error: {str(e)}"