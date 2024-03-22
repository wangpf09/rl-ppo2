import torch

from algorithm import ppo2
from game import game_env


def eval_model(env_id):
    env = game_env.init_env(env_id, True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ppo = ppo2.PPO2(state_dim, action_dim, 64, device)
    ppo.actor.load_state_dict(torch.load('./models/best_actor.pth'))
    ppo.critic.load_state_dict(torch.load('./models/best_critic.pth'))

    ppo.actor.eval()
    ppo.critic.eval()
    with torch.no_grad():
        play_in_round(env, ppo, 10)


def play_in_round(env, ppo, max_episodes):
    state = env.reset(seed=42)[0]
    count = 0
    total_reward = 0
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(ppo.device)
        action = ppo.get_action(state_tensor)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        if terminated or truncated:
            observation, info = env.reset()
            count += 1
            print(f"count: {count} reward: {total_reward}")
            total_reward = 0

        if count > max_episodes:
            return
        state = observation


if __name__ == '__main__':
    _env_id = "LunarLander-v2"
    eval_model(_env_id)
