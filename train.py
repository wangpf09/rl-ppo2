import torch

import eval
from algorithm import ppo2
from game import game_env


def train_model(env, ppo, num_episodes):
    play_in_round(env, ppo, num_episodes)
    eval.play_in_round(env, ppo, 1)


def play_in_round(env, ppo, max_episodes):
    state = env.reset(seed=42)[0]
    total_reward = 0
    best_reward = float('-inf')
    save_interval = 300
    epoch = 0
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(ppo.device)
        action = ppo.get_action(state_tensor)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        ppo.collect_data(state_tensor, action, reward)
        if terminated or truncated:
            ppo.train(epoch)
            ppo.total_rewards.append(total_reward)
            ppo.clean_collect_data()
            observation, info = env.reset()
            ppo.writer.add_scalar('total-reward', total_reward, epoch)
            epoch += 1
            print(f"epoch: {epoch} reward: {total_reward}")

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(ppo.actor.state_dict(), './models/best_actor.pth')
                torch.save(ppo.critic.state_dict(), './models/best_critic.pth')
                print(f"Epoch [{epoch}/{max_episodes}] - Saved best model with reward: {total_reward:.2f}")

            if epoch % save_interval == 0:
                torch.save(ppo.actor.state_dict(), f'./models/actor_{epoch}.pth')
                torch.save(ppo.critic.state_dict(), f'./models/critic_{epoch}.pth')
                print(f"Epoch [{epoch}/{max_episodes}] - Saved model at epoch {epoch}")

            total_reward = 0
        if epoch > max_episodes:
            return
        state = observation


if __name__ == '__main__':
    env_id = "LunarLander-v2"
    _env = game_env.init_env(env_id, False)
    state_dim = _env.observation_space.shape[0]
    action_dim = _env.action_space.n
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _ppo = ppo2.PPO2(state_dim, action_dim, 64, device)

    train_model(_env, _ppo, 3000)

    _ppo.writer.close()
