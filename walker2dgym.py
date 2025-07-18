import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import time
import numpy as np
import random
import mujoco
import mujoco_viewer

train_env = gym.make("Walker2d-v5")
mjmodel = train_env.unwrapped.model
model = PPO(
    "MlpPolicy", 
    train_env, 
    gamma=0.99, 
    n_steps=8192, 
    batch_size=256, 
    learning_rate=1e-4,
    device="cpu", 
    tensorboard_log="./ppo_tensorboard/")
model.learn(total_timesteps=3000000)
model.save("ppo_walker_3M")

train_env.close()


data = mujoco.MjData(mjmodel)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(mjmodel, data)

# simulate and render
for _ in range(10000):
    if viewer.is_alive:
        position = data.qpos.flatten()[1:]
        velocity = np.clip(data.qvel.flatten(), -10, 10)
        obs = np.concatenate((position, velocity)).ravel()        
        action, _states = model.predict(obs)
        data.ctrl[:] = action
        mujoco.mj_step(mjmodel, data)
        viewer.render()
    else:
        break

# close
viewer.close()