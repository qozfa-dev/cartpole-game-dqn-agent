# cartpole-game-dqn-agent

This project implements a Deep Q-Network (DQN) to solve the classic CartPole-v1 environment using PyTorch and Gymnasium. The agent learns to balance a pole on a moving cart through reinforcement learning.

## Features
- Deep Q-Learning with experience replay
- Epsilon-greedy exploration strategy
- Model training with PyTorch
- Training progress visualization using matplotlib
- Simple modular code structure

## Setup
### 1. Clone the repository:
```bash
git clone https://github.com/qozfa-dev/cartpole-game-dqn-agent.git
cd cartpole-game-dqn-agent
```

### 2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```
If you don't have requirements.txt, install manually:
```bash
pip install gymnasium torch matplotlib numpy
```

## Usage
### Train the agent:
```bash
python train.py
```
- Training will run for 1000 episodes by default.
- A reward plot will be saved as training_progress.png (inside the /data folder).

### Output
- Console Logs: Total reward and epsilon per episode.
- Graph: A plot showing reward trends over time.

### License
MIT
