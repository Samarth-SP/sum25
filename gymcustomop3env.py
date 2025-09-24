import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import time
import numpy as np
import random
import mujoco
import mujoco_viewer

from gymnasium.envs.registration import register


register(
    id="QuadEnv-v0",
    entry_point="quadruped:QuadEnv",
)

class ent_callback(BaseCallback):
    def __init__(self, i_ent, f_ent, steps, verbose = 0):
        super().__init__(verbose)
        self.i_ent = i_ent
        self.f_ent = f_ent
        self.steps = steps
    def _on_step(self):
        self.model.ent_coef = self.i_ent + (self.f_ent - self.i_ent) * min((self.num_timesteps/self.steps), 1.0)
        self.logger.record("rollout/ent_coef", self.model.ent_coef)
        return True
timesteps = 10000000
lr_sched = get_linear_fn(3e-4, 5e-5, 1.0)
cb = ent_callback(1e-3, 3e-4, timesteps)
vec_env = make_vec_env("QuadEnv-v0", n_envs=8)
vec_env.reset()
model = PPO(
    policy="MlpPolicy",
    env=vec_env, 
    n_steps=4096, 
    batch_size=256, 
    learning_rate=lr_sched,
    ent_coef=1e-3,
    target_kl=0.13,
    device="cpu", 
    tensorboard_log="./ppo_op3/")
# model = SAC(
#     policy="MlpPolicy",
#     env=train_env, 
#     ent_coef=0.01,
#     tensorboard_log='./ql_op3',
#     device="cpu")
model.learn(total_timesteps=timesteps, callback=cb)
model.save(f"quad_ppo_5M")


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
# test_env.close()
