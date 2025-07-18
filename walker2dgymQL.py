import gymnasium as gym
from stable_baselines3 import SAC
import time
import numpy as np
import random
import mujoco
import mujoco_viewer

train_env = gym.make("Walker2d-v5")
train_env.reset()
mjmodel = train_env.unwrapped.model
model = SAC(
    policy="MlpPolicy",
    env=train_env, 
    device="cpu", 
    tensorboard_log="./ql_tensorboard/")
model.learn(total_timesteps=300000)
model.save("QL_walker_300K")
train_env.close()

test_env = gym.make("Walker2d-v5", render_mode="human")
test_env.reset()
for i in range(2):
    obs, info = test_env.reset()
    ep_end = False
    while not ep_end:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        if terminated or truncated:
            obs, info = test_env.reset()
        ep_end = terminated or truncated
test_env.close()
