import gym
import matplotlib.pyplot as plt
from agent import DQNAgent


def train_dqn(agent, env, episodes=1000):
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(
            f"Episode {episode+1}/{episodes} - Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

        if (episode + 1) % 100 == 0:
            plt.plot(total_rewards)
            plt.savefig('results/training_progress.png')

    return total_rewards


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = DQNAgent(input_dim=4, output_dim=env.action_space.n)

    train_dqn(agent, env, episodes=1000)
