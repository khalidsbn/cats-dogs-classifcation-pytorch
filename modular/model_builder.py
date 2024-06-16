
import torch
import torch.nn as nn

class CNN(nn.Module):
  """Creates CNN architecture."""
    
  def __init__(self):
    super(CNN, self).__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.layer2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.layer3 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.fc1 = nn.Linear(3 * 3 * 64, 10)
    self.dropout = nn.Dropout(0.5)
    self.fc2 = nn.Linear(10, 2)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = x.view(x.size(0), -1)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x
