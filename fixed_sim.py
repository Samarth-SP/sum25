import gymnasium as gym
from stable_baselines3 import PPO, SAC
from scipy.interpolate import CubicSpline
import time
import numpy as np
import random
import mujoco
import mujoco_viewer
import math
from quadruped import QuadEnv

env = QuadEnv()
mjmodel = env.unwrapped.model
data = mujoco.MjData(mjmodel)

# S1 = [ 0.2, 0.2, 0.25, 0.4, -0.4, -0.1, -0.75, -0.75 ]
# S2 = [ -0.6, 0.6, 0.1, 0.0, 0.0, 0.05, 0.25, 0.25 ]
# S3 = [ -0.4, 0.4, 0.1, -0.2, -0.2, -0.25, -0.75, -0.75 ]
# S4 = [ 0.0, 0.0, -0.05, 0.6, -0.6, -0.1, -0.25, -0.25 ]
# S5 = [ 0.2, 0.2, 0.25, 0.4, -0.4, -0.1, -0.75, -0.75 ]
# waypoints = np.vstack([S1, S2, S3, S4, S5])
# t_nodes = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

# spline = CubicSpline(t_nodes, waypoints, bc_type='periodic', axis=0)
# create the viewer object
viewer = mujoco_viewer.MujocoViewer(mjmodel, data)
# simulate and render
for _ in range(10000):
    if viewer.is_alive:
        chatarr = S3 = [ -0.4, 0.4, 0.1, -0.2, -0.2, -0.25, 0.25, 0.25 ]
#spline(0.001*_)
        # data.qpos[17] = chatarr[0] #lhippitch
        # data.qpos[18] = chatarr[1] #lknee
        # data.qpos[19] = chatarr[2] #lanklepitch
        # data.qpos[23] = chatarr[3] #rhippitch
        # data.qpos[24] = chatarr[4] #rknee
        # data.qpos[25] = chatarr[5] #ranklepitch
        # data.qpos[9] = chatarr[6] #lshopitch
        # data.qpos[12] = chatarr[7] #rshopitch
        mujoco.mj_step(mjmodel, data)
        print(data.qpos)
        viewer.render()
    else:
        break
# [ 0.2, 0.2, 0.5, 0.4, -0.4, -0.1, -0.5, -0.5 ] right forward planted
# [ -0.6, 0.6, 0.1, 0.0, 0.0, 0.05, 0.25, 0.25 ] left forward in air
# [ -0.4, 0.4, 0.1, -0.2, -0.2, -0.5, -0.5, -0.5 ] left forward planted
# [ 0.0, 0.0, -0.05, 0.6, -0.6, -0.1, -0.25, -0.25 ] right forward in air
# [ 0.2, 0.2, 0.5, 0.4, -0.4, -0.1, -0.5, -0.5 ] right forward planted
# close
viewer.close()  
#np.savetxt('joint.csv', d, delimiter=", ")
# rewsum = 0
# forrewsum = 0
# test_env = gym.make("op3alt", render_mode="human")
# test_env.reset()
# for i in range(2):
#     obs, info = test_env.reset()
#     ep_end = False
#     while not ep_end:
#         action, _states = model.predict(obs)
#         obs, reward, terminated, truncated, info = test_env.step(action)
#         #print(info)
#         test_env.render()
#         if terminated:
#             obs, info = test_env.reset()
#         ep_end = terminated or truncated
#         rewsum += reward
#         forrewsum += info['reward_forward']
# test_env.close()

# print(forrewsum, rewsum)