import gymnasium as gym
from stable_baselines3 import PPO
import time
import numpy as np
import random


train_env = gym.make("LunarLander-v3")
print(train_env.metadata)
model = PPO("MlpPolicy", train_env, gamma=0.9, device="cpu")
model.learn(total_timesteps=1000000)
train_env.close()
env = gym.make("LunarLander-v3", render_mode="human")
for i in range(2):
    obs, info = env.reset()
    ep_end = False
    while not ep_end:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
        ep_end = terminated or truncated
env.close()
'''
-set lambda to 0.9, robot barely moves, likely overvaluing reward for moving slow
-under 100000 time_steps leads to non-solutions
-
'''
