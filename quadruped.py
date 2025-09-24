__credits__ = ["Kallinteris-Andreas"]

from typing import Dict, Tuple, Union

import numpy as np
import math

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from scipy.interpolate import CubicSpline


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class QuadEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "/Users/samarth/miniconda3/lib/python3.12/site-packages/mujoco_playground/external_deps/mujoco_menagerie/anybotics_anymal_b/anymal_b.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.5,
        ctrl_cost_weight: float = 5e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.30, 0.70),
        healthy_angle_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range


        self._reset_noise_scale = reset_noise_scale
        self.step_count = 0
        self.phi = 0
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def orientation(self, data):
        id = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        r = data.ximat[1]
        rmat = np.reshape(r, (3,3)) 
        relative = rmat @ id.T
        relAngle = (np.trace(relative) - 1) / 2
        relAngle = np.clip(relAngle, -1.0, 1.0)
        angle = np.arccos(relAngle)
        return 0.25*np.square(angle) + np.abs(angle)
    
    def torque(self, data):
        return np.sum(np.abs(data.actuator_force))
    
    def energy(self, data):
        return np.sum(np.abs(data.qvel[6:]) * np.abs(data.actuator_force))
    
    
    def zvel(self, data):
        return np.abs(data.qvel[2])
    
    @property
    def is_healthy(self):
        z = self.data.qpos[2]
        min_z, max_z = self._healthy_z_range
        healthy_z = min_z < z < max_z
        is_healthy = healthy_z
        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        self.step_count+=1
        x_position_before = self.data.qpos[0]
        action = action + np.random.normal(0, 0.5, size=action.shape)
        action = np.clip(action, -1, 1)
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        ypos = self.data.qpos[1]
        x_velocity = (x_position_after - x_position_before) / self.dt
        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, ypos, self.data)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[2] - self.init_qpos[2],
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity, ypos, data):
        forward_reward_weight: float = 4.25
        forward_exp_weight: float = 6.0
        healthy_reward_weight: float = 0.8

        orientation_cost_weight: float = 2.75
        sideways_cost_weight: float = 1.0
        torque_cost_weight: float = 0.0002
        energy_cost_weight: float = 0.0001
        zvel_cost_weight: float = 0.1

        forward_reward = forward_reward_weight * np.exp(-forward_exp_weight*(x_velocity - 0.6)**2)
        healthy_reward = healthy_reward_weight * self.healthy_reward
        rewards = forward_reward + healthy_reward

        orientation_cost = orientation_cost_weight * self.orientation(data)
        sideways_cost = sideways_cost_weight * np.abs(ypos)
        torque_cost = torque_cost_weight * self.torque(data)
        energy_cost = energy_cost_weight * self.energy(data)
        zvel_cost = zvel_cost_weight * self.zvel(data)
        costs =  orientation_cost + sideways_cost + torque_cost + energy_cost + zvel_cost

        reward = rewards - costs
        if(not self.is_healthy):
            reward -= (-200/(1+np.exp(-0.006*(self.step_count-300)))+200)
            self.step_count = 0

        reward_info = {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "cost_orientation": -orientation_cost,
            "cost_sideways": -sideways_cost,
            "cost_torque": -torque_cost,
            "cost_energy": -energy_cost,
            "cost_zvel": -zvel_cost,
        }

        return reward, reward_info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        self.step_count = 0
        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }
