import torch
import torch.nn as nn
import torch.nn.functional as F

# class PolicyNetwork(nn.Module):

#     def __init__(self, input_channels=4, num_actions=5):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_channels, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.ReLU()
#         )
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32 * 10 * 10, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_actions)
#         )

#     def forward(self, x):
#         return self.fc(self.conv(x))
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_channels=4, num_actions=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d((4, 4))  # ⬅️ 强制把 feature map 变为 4x4
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.output = nn.Linear(128, num_actions)

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)  # 🔁 统一特征尺寸
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.output(x)

