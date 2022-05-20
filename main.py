import numpy as np
import ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray import tune

from env import Env

CONFIG = {
    "config": {
        "env": "quadcopter",
        "env_config": {
            "sim": {
                "dt": 0.01,
                "max_t": 5.,
            },
            "init": {
                "pos": np.vstack((2, 2, -2)),
                "vel": np.zeros((3, 1)),
                "R": np.eye(3),
                "omega": np.zeros((3, 1)),
            },
        },
        "framework": "tf2",
        "eager_tracing": True,
        "num_gpus": 0,
        "num_workers": 5,
        # "num_envs_per_worker": 10,
        # "lr": 0.0001,
        # "gamma": 0.999,
        "lr": tune.grid_search([0.001, 0.0005, 0.0001]),
        "gamma": tune.grid_search([0.9, 0.99, 0.999]),
    },
    "stop": {
        "training_iteration": 1000,
    },
    "local_dir": "./ray_results",
    "checkpoint_freq": 100,
    "checkpoint_at_end": True,
}

def debug():
    env = Env(CONFIG['config']['env_config'])
    obs = env.reset()
    while True:
        action = np.array([1, 1, 1, 1])
        obs, reward, done, _ = env.step(action)
        if done:
            break
    env.close()


def train():
    ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    trainer = tune.run(ppo.PPOTrainer, **CONFIG)
    ray.shutdown()


def main():
    register_env("quadcopter", lambda env_config: Env(env_config))

    train()


if __name__ == "__main__":
    main()
    # debug()


    


