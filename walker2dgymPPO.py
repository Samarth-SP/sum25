import gymnasium as gym
from stable_baselines3 import PPO
import time
import numpy as np
import random
import mujoco
import mujoco_viewer

train_env = gym.make("Walker2d-v5")
train_env.reset()
mjmodel = train_env.unwrapped.model
model = PPO(
    policy="MlpPolicy",
    env=train_env, 
    gamma=0.99, 
    n_steps=4096, 
    batch_size=256, 
    learning_rate=1e-4,
    device="cpu", 
    tensorboard_log="./ppo_tensorboard/")
model.learn(total_timesteps=3000000)
model.save("ppo_walker_3M")
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
