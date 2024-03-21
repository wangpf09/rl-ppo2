import train
from algorithm import ppo2
from game import game_env

if __name__ == '__main__':
    env_id = "CartPole-v1"
    env = game_env.init_env(env_id, False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo2 = ppo2.PPO2(state_dim, action_dim, 64)

    train.train(env, ppo2, 10)
