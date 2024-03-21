import time

import gym


def init_env(env_id, record_video):
    def thunk():
        if record_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f'./video/{env_id}.{int(time.time())}')
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk()
