import gymnasium as gym
from stable_baselines3 import SAC, PPO
import time
import numpy as np
import random
import mujoco
import mujoco_viewer
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("op3alt", n_envs=8)
vec_env.reset()


model = PPO.load("op3_ppo_v1test1", env=vec_env)
model.ent_coef = 0
model.learning_rate = 1e-5
model.learn(total_timesteps=10000000)
model.save(f"op3_ppo_v1test3")


# test_env = gym.make("op3", render_mode="human")
# test_env.reset()
# for i in range(2):
#     obs, info = test_env.reset()
#     ep_end = False
#     while not ep_end:
#         action, _states = model.predict(obs)
#         obs, reward, terminated, truncated, info = test_env.step(action)
#         test_env.render()
#         if terminated or truncated:
#             obs, info = test_env.reset()
#         ep_end = terminated or truncated
# test_env.close()1
