import gymnasium as gym
from stable_baselines3 import SAC, PPO
import time
import numpy as np
import random
import mujoco
import mujoco_viewer

train_env = gym.make("op3alt")
train_env.reset()
mjmodel = train_env.unwrapped.model
model = PPO(
    policy="MlpPolicy",
    env=train_env, 
    gamma=0.98, 
    n_steps=4096, 
    batch_size=256, 
    learning_rate=1e-4,
    ent_coef=1e-3,
    device="cpu", 
    tensorboard_log="./ppo_op3/")
model.set_env(train_env)
# model = SAC(
#     policy="MlpPolicy",
#     env=train_env, 
#     ent_coef=0.01,
#     tensorboard_log='./ql_op3',
#     device="cpu")

model.learn(total_timesteps=10000000)
model.save(f"op3_ppo_v1")


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
