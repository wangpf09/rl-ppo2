import torch

from algorithm import ppo2
from game import game_env


def train_model(env, ppo, num_episodes):
    play_in_round(env, ppo, num_episodes)


def play_in_round(env, ppo, max_episodes):
    state = env.reset(seed=42)[0]
    count = 0
    total_reward = 0
    best_time = 0
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(ppo.device)
        action = ppo.get_action(state_tensor)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        ppo.collect_data(state_tensor, action, reward)
        if terminated or truncated:
            ppo.train(count)
            ppo.total_rewards.append(total_reward)
            ppo.clean_collect_data()
            observation, info = env.reset()
            ppo.writer.add_scalar('total-reward', total_reward, count)
            count += 1
            print(f"count: {count} reward: {total_reward}")

            if total_reward == 500:
                best_time += 1
            else:
                best_time = 0
            if best_time > 5:
                ppo.save_model(count, True)
            total_reward = 0

        if count > max_episodes:
            return
        state = observation


if __name__ == '__main__':
    env_id = "CartPole-v1"
    _env = game_env.init_env(env_id, False)
    state_dim = _env.observation_space.shape[0]
    action_dim = _env.action_space.n
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _ppo = ppo2.PPO2(state_dim, action_dim, 64, device)

    train_model(_env, _ppo, 1000)

    _ppo.writer.close()
