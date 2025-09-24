# train_op3.py
"""
Minimal training script that uses Brax PPO to train OP3Env.
"""

import functools
import jax
import jax.numpy as jnp

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train

from Coding.summer25.quadruped import OP3Env


def progress_fn(step: int, metrics: dict):
    if step % 100 == 0 and step > 0:
        print(f"step={step:8d}  eval/episode_reward={metrics.get('eval/episode_reward', float('nan')):.3f}")


def main():
    rng = jax.random.PRNGKey(0)

    # Create a single env instance (PipelineEnv). Brax trainer will vectorize internally.
    env = OP3Env(xml_file="robotis_op3/op3.xml", frame_skip=4)

    # small network factory example
    make_networks = functools.partial(
        ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(128, 128)
    )

    # partial training config
    train_fn = functools.partial(
        ppo_train,
        num_timesteps=50_000,     # shorten for quick sanity-run
        num_envs=32,
        episode_length=200,
        unroll_length=16,
        num_minibatches=8,
        num_updates_per_batch=4,
        learning_rate=3e-4,
        discounting=0.97,
        entropy_cost=1e-3,
        normalize_observations=True,
        reward_scaling=1.0,
        network_factory=make_networks,
        seed=0,
    )

    # run training
