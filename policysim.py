import gymnasium as gym
from stable_baselines3 import PPO, SAC
import time
import numpy as np
import random
import mujoco
import mujoco_viewer
import math
from quadruped import QuadEnv

from gymnasium.envs.registration import register


register(
    id="QuadEnv-v0",
    entry_point="quadruped:QuadEnv",
)

model = PPO.load("quad_ppo_5M")
env = gym.make("QuadEnv-v0")
mjmodel = env.unwrapped.model
data = mujoco.MjData(mjmodel)

for i in range(mjmodel.ngeom):
    name_addr = mjmodel.name_geomadr[i]
    name = mjmodel.names[name_addr:].split(b'\x00', 1)[0].decode('utf-8')
    print(f"{i}: {name}")

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(mjmodel, data)
# simulate and render
energy = 0
d = []
for _ in range(10000):  
    if viewer.is_alive:
        position = data.qpos.flatten()[1:]
        velocity = np.clip(data.qvel.flatten(), -10, 10)
        obs = np.concatenate((position, velocity)).ravel()        
        if(not _ % 4):
            action, _states = model.predict(obs) #every four steps
        data.ctrl[:] = action
        energy += np.sum(np.abs(data.qvel[6:]) * np.abs(data.actuator_force)) * 0.002
        print(data.qpos)
        mujoco.mj_step(mjmodel, data)
        viewer.render() #every six steps
        d.append(data.qpos[6:].copy())
    else:
        break

# close
viewer.close()  
#np.savetxt('joint.csv', d, delimiter=", ")
# test_env = gym.make("QuadEnv-v0", render_mode="human")
# test_env.reset()
# for i in range(2):
#     obs, info = test_env.reset()
#     ep_end = False
#     while not ep_end:
#         action, _states = model.predict(obs)
#         obs, reward, terminated, truncated, info = test_env.step(action)
#         #print(info)
#         test_env.render()
#         print(obs)
#         if terminated:
#             obs, info = test_env.reset()
#         ep_end = terminated or truncated
#         print(info)
# test_env.close()

# print(forrewsum, rewsum)