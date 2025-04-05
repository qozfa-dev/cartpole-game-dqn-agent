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

    # Selects an action based on epsilon-greedy policy
    def select_action(self, state):
        if random.random() < self.epsilon:
            # Exploration mode is entered if less than epsilon value
            # In CartPole, 0 = Left, 1 = Right
            return np.random.choice([0, 1])
        else:
            # Exploitation mode is entered otherwise
            # convert the state to a pytorch tensor
            state_tensor = torch.tensor(state, dtype=torch.float32)
            # Get Q-values from the Q-network
            q_values = self.q_network(state_tensor)
            # Return the action with the highest Q-value
            return torch.argmax(q_values).item()

    # Trains the Q-network using experiences from the replay buffer
    def train(self):
        # Ensure there are enough experiences in the buffer before training
        if self.replay_buffer.size() < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to torch tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Get Q-values for current states
        q_values = self.q_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get Q-values for next states (using the target network)
        next_q_values = self.target_network(next_states).max(1)[0]

        # Calculate target Q-values using the Bellman equation
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute the loss (mean squared error)
        loss = torch.nn.MSELoss()(current_q_values, target_q_values)

        # Backpropagation: Update Q-network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon (decrease exploration over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Periodically copy weights from Q-network to target network
        if self.epsilon % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
