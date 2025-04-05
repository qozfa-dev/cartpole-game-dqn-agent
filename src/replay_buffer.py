import random
from collections import deque

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
