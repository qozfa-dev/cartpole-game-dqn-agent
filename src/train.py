import gymnasium as gym
import matplotlib.pyplot as plt
from DqnAgent import DqnAgent


def train_dqn(agent, env, episodes=1000):
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            # Render the environment to see the CartPole in action
            env.render()

        total_rewards.append(total_reward)
        print(
            f"Episode {episode+1}/{episodes} - Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

        # Save model periodically
        if (episode + 1) % 100 == 0:
            plt.plot(total_rewards)
            plt.savefig('training_progress.png')

    return total_rewards


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = DqnAgent(input_dim=4, output_dim=env.action_space.n)

    train_dqn(agent, env, episodes=1000)
