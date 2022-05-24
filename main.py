import numpy as np
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from scipy.spatial.transform import Rotation as rot
from pathlib import Path

import fym
from env import Env

CONFIG = {
    "config": {
        "env": "quadcopter",
        "env_config": {
            "sim": {
                "dt": 0.01,
                "max_t": 3.,
            },
            "init": {
                "pos": np.zeros((3, 1)),
                "vel": np.zeros((3, 1)),
                "R": rot.from_euler(
                    "ZYX", np.deg2rad([10, 10, 10])
                ).as_matrix(),
                "omega": np.zeros((3, 1)),
            },
        },
        "framework": "torch",
        # "eager_tracing": True,
        "num_gpus": 0,
        "num_workers": 5,
        # "num_envs_per_worker": 10,
        "lr": 0.0001,
        "gamma": 0.99,
        # "lr": tune.grid_search([0.001, 0.0005, 0.0001]),
        # "gamma": tune.grid_search([0.9, 0.99, 0.999]),
    },
    "stop": {
        "training_iteration": 10000,
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
    trainer = tune.run(ppo.PPOTrainer, **CONFIG)
    logdir = trainer.get_best_logdir(
        metric='episode_reward_mean',
        mode='max'
    )
    checkpoint_paths = trainer.get_trial_checkpoints_paths(logdir)
    return checkpoint_paths


@ray.remote(num_cpus=5)
def sim(checkpoint_path):
    parent_path = Path(checkpoint_path).parent
    data_path = Path(parent_path, "sim_data.h5")

    CONFIG['explore'] = False
    agent = ppo.PPOTrainer(config=CONFIG['config'], env=Env)
    agent.restore(checkpoint_path)

    env = Env(CONFIG['config']['env_config'])
    env.logger = fym.Logger(data_path)

    obs = env.reset(random=False)
    while True:
        action = agent.compute_single_action(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break
    env.close()


def evaluate(checkpoint_paths):
    futures = [sim.remote(path[0]) for path in checkpoint_paths]
    ray.get(futures)


def main():
    ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    register_env("quadcopter", lambda env_config: Env(env_config))

    checkpoint_paths = train()
    evaluate(checkpoint_paths)
    ray.shutdown()

    sim_data = fym.logging.load(
        Path(Path(checkpoint_paths[-1][0]).parent, 'sim_data.h5')
    )
    breakpoint()
    


if __name__ == "__main__":
    main()
    # debug()


    


