import gymnasium as gym
from stable_baselines3 import PPO, SAC
import time
import numpy as np
import random
import mujoco
import mujoco_viewer
import math

model = PPO.load("op3_ppo_bestalt")
env = gym.make("op3alt")
mjmodel = env.unwrapped.model
data = mujoco.MjData(mjmodel)

# for i in range(mjmodel.ngeom):
#     name_addr = mjmodel.name_geomadr[i]
#     name = mjmodel.names[name_addr:].split(b'\x00', 1)[0].decode('utf-8')
#     print(f"{i}: {name}")

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
        qw, qx, qy, qz = data.qpos[4:8]
        print(data.geom_xpos[50][1] - data.geom_xpos[37][1])
        # print(qw-1, qx, qy-0.2, qz)
        # print(data.contact.geom)
        # print(np.sum(np.abs(data.qfrc_actuator)))
        mujoco.mj_step(mjmodel, data)
        viewer.render()
    else:
        break

# close
viewer.close()  

# test_env = gym.make("op3alt", render_mode="human")
# test_env.reset()
# for i in range(2):
#     obs, info = test_env.reset()
#     ep_end = False
#     while not ep_end:
#         action, _states = model.predict(obs)
#         obs, reward, terminated, truncated, info = test_env.step(action)
#         print(info)
#         test_env.render()
#         if terminated or truncated:
#             obs, info = test_env.reset()
#         ep_end = terminated or truncated
# test_env.close()