import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc3(x)

if __name__ == '__main__':
    state_size = 12
    action_size = 2
    net = DQN(state_size, action_size)
    state = torch.randn(10, state_size)
    output = net(state)
    print("hello")
    print(output)

