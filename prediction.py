import torch
import torch.nn as nn


class ANN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(4,16),
            nn.ReLU(),

            nn.Linear(16,8),
            nn.ReLU(),

            nn.Linear(8,1)

        )

    def forward(self,x):

        return self.net(x)


model = ANN()

model.load_state_dict(torch.load("best_model.pt"))

model.eval()


def predict_power(data):

    data = torch.FloatTensor(data)

    prediction = model(data)

    return prediction.item()