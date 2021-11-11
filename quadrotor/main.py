import numpy as np
import numpy.random as random
import ray
from ray import tune
from ray.rllib.agents import ddpg
from ray.rllib.agents import ppo
import gym
from pathlib import Path

import fym
from fym.core import BaseEnv, BaseSystem
from postProcessing import plot_TwoDimTwoPointMass
from dynamics import EnvTwoDimPointMass


def train():
    cfg = fym.config.load(as_dict=True)
    analysis = tune.run(ppo.PPOTrainer, **cfg)
    parent_path = Path(analysis.get_last_checkpoint(
        metric="episode_reward_mean",
        mode="max"
    )).parent.parent
    checkpoint_paths = analysis.get_trial_checkpoints_paths(
        trial=str(parent_path)
    )
    return checkpoint_paths


@ray.remote(num_cpus=6)
def sim(initial, checkpoint_path, env_config, num=0):
    env = EnvTwoDimPointMass(env_config)
    agent = ppo.PPOTrainer(env=EnvTwoDimPointMass, config={"explore": False})

    agent.restore(checkpoint_path)
    parent_path = Path(checkpoint_path).parent
    data_path = Path(parent_path, f"test_{num+1}", "env_data.h5")
    plot_path = Path(parent_path, f"test_{num+1}")
    env.logger = fym.Logger(data_path)

    obs = env.reset(initial)
    total_reward = 0
    while True:
        action = agent.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    env.close()
    plot_TwoDimTwoPointMass(plot_path, data_path)


def validate(parent_path):
    _, info = fym.logging.load(
        Path(parent_path, 'checkpoint_paths.h5'),
        with_info=True
    )
    checkpoint_paths = info['checkpoint_paths']
    initials = []
    print("Making initials...")
    random.seed(0)
    while True:
        tmp = 20 * (2 * np.random.rand(4,1) -1)
        if np.all([
                np.sqrt(np.sum(tmp[:2, :]**2, axis=0)) < 10,
                np.sqrt(np.sum(tmp[:2, :]**2, axis=0)) > 5,
                np.sqrt(np.sum(tmp[2:, :]**2, axis=0)) < 3,
        ], axis=0):
            initials.append(tmp)
        if len(initials) == 5:
            break
    fym.config.update({"config.env_config.max_t": 20})
    env_config = ray.put(fym.config.load("config.env_config", as_dict=True))
    print("Validating...")
    futures = [sim.remote(initial, path[0], env_config, num=i)
               for i, initial in enumerate(initials)
               for path in checkpoint_paths]
    ray.get(futures)

def plot_data(parent_path_list):
    for parent_path in parent_path_list:
        _, info = fym.logging.load(
            Path(parent_path, 'checkpoint_paths.h5'),
            with_info=True
        )
        checkpoint_paths = info['checkpoint_paths']
        # for checkpoint_data in checkpoint_paths:
        checkpoint_path = Path(checkpoint_paths[-1][0]).parent
        test_path_list = [x for x in checkpoint_path.iterdir() if x.is_dir()]
        # for i in range(len(test_path_list)):
        data_path = Path(test_path_list[-1], "env_data.h5")
        plot_path = Path(test_path_list[-1])
        print("Ploting", str(plot_path))
        plot_rllib_test(plot_path, data_path)

def main():
    fym.config.reset()
    fym.config.update({
        "config": {
            "env": EnvTwoDimPointMass,
            "env_config": {
                "dt": 0.01,
                "max_t": 10.,
                "solver": "rk4"
            },
            "num_gpus": 0,
            "num_workers": 4,
            # "num_envs_per_worker": 50,
            "lr": 0.0001,
            "gamma": 0.99,
            # "lr": tune.grid_search([0.001, 0.003, 0.0001]),
            # "gamma": tune.grid_search([0.9, 0.99, 0.999])
            # "actor_lr": tune.grid_search([0.001, 0.003, 0.0001]),
            # "critic_lr": tune.grid_search([0.001, 0.003, 0.0001]),
            # "actor_lr": 0.001,
            # "critic_lr": 0.0001,
            # "gamma": tune.grid_search([0.9, 0.99, 0.999, 0.9999]),
            # "exploration_config": {
            #     "random_timesteps": 10000,
            #     "scale_timesteps": 100000,
            # },
        },
        "stop": {
            "training_iteration": 3,
        },
        "local_dir": "./ray_results",
        "checkpoint_freq": 2,
        "checkpoint_at_end": True,
    })
    checkpoint_paths = train()
    parent_path = "/".join(checkpoint_paths[0][0].split('/')[0:-3])
    checkpoint_logger = fym.logging.Logger(
        Path(parent_path, 'checkpoint_paths.h5')
    )
    checkpoint_logger.set_info(checkpoint_paths=checkpoint_paths)
    checkpoint_logger.set_info(config=fym.config.load(as_dict=True))
    checkpoint_logger.close()
    return parent_path


if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    ## To train, validate, and make figure
    parent_path = main()
    ## To validate and make figure
    # parent_path = './ray_results//'
    validate(parent_path)
    ray.shutdown()

    # To only Make Figure
    # pathes = [
    #     './ray_results/PPO_2021-10-30_11-14-17/',
    #     './ray_results/PPO_2021-10-30_04-39-15/',
    #     './ray_results/PPO_2021-10-30_17-48-39/',
    #     './ray_results/PPO_2021-10-29_00-16-02/',
    #     './ray_results/PPO_2021-10-30_19-14-30/',
    #     './ray_results/PPO_2021-10-30_20-59-25/',
    #     './ray_results/PPO_2021-10-31_03-13-36/',
    #     './ray_results/PPO_2021-10-30_07-49-32/',
    #     './ray_results/PPO_2021-10-31_08-35-38/',
    # ]
    # plot_data(pathes)




