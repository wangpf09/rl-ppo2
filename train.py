import torch
from game import game_env
import matplotlib.pyplot as plt


def train(env, ppo, num_episodes):
    for _ in range(num_episodes):
        play_in_round(env, ppo, 1000)

        # Plotting rewards history
    plt.plot(ppo.all_rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    env_id = "CartPole-v1"
    env = game_env.init_env(env_id, True)
    play_in_round(env, ppo, 1)


def play_in_round(env, ppo, max_episodes):
    state = env.reset(seed=42)[0]
    count = 0
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = ppo.get_action(state_tensor)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        ppo.collect_data(state_tensor, action, reward)
        if terminated or truncated:
            ppo.train()
            ppo.clean_collect_data()
            observation, info = env.reset()
            count += 1
            print(f"count: {count} reward: {reward}")
        if count == max_episodes:
            print(f"已经运行过{count}次")
            return
        state = observation
