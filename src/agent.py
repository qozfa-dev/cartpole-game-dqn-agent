import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# QNetwork class is used to approximate the Q-values using in Q-learning


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # Multiple layers of transformations
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        # Apply ReLU for non-linearity (negative values become 0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x    # Returns the Q-values


# ReplayBuffer stores the agent's experiences so the agent can sample to learn from past experiences


class ReplayBuffer:
    def __init__(self, max_size):
        # Uses a deque to store experiences, old experiences are discarded when the buffer is full
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        # Adds a new experience (state, action, reward, next_state, done) to the buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly samples a batch of experiences from the buffer
        return random.sample(self.buffer, batch_size)

    def size(self):
        # Returns how many experiences are stored (size of buffer)
        return len(self.buffer)
