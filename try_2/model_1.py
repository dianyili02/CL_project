import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):

    def __init__(self, input_channels=4, num_actions=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.fc(self.conv(x))
