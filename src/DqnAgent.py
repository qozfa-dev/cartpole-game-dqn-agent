import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque
from agent import QNetwork
from replay_buffer import ReplayBuffer


class DqnAgent:
    # Initialises the DQN Agent with parameters
    def __init__(self, input_dim, output_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.epsilon_min = epsilon_min  # Minimum epsilon value to stop exploration

        # Initialize the Q-network and target network
        # Q-network predicts Q-values for actions.
        self.q_network = QNetwork(input_dim, output_dim).float()

        # Target network is a copy of the Q-network used to generate stable Q-values
        self.target_network = QNetwork(input_dim, output_dim).float()

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Use Adam optimizer for updating Q-network weights
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate)

        # Replay buffer for storing experiences
        self.replay_buffer = ReplayBuffer(max_size=10000)
        self.batch_size = 64  # Batch size for training
