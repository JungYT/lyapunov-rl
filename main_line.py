import numpy as np
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from scipy.spatial.transform import Rotation as rot
from pathlib import Path
from matplotlib import pyplot as plt

import fym
from env import EnvLine

CONFIG = {
    "config": {
        "env": "line",
        "env_config": {
            "sim": {
                "dt": 0.01,
                "max_t": 5.,
            },
            "init": {
                "pos": np.vstack([5.]),
                "vel": np.vstack([1.]),
            },
        },
        "framework": "torch",
        # "eager_tracing": True,
        "num_gpus": 0,
        "num_workers": 5,
        # "num_envs_per_worker": 10,
        "lr": 0.0001,
        "gamma": 0.9,
        # "lr": tune.grid_search([0.001, 0.0005, 0.0001]),
        # "gamma": tune.grid_search([0.9, 0.99, 0.999]),
    },
    "stop": {
        "training_iteration": 2000,
    },
    "local_dir": "./ray_results/line",
    "checkpoint_freq": 100,
    "checkpoint_at_end": True,
}

def debug():
    env = EnvLine(CONFIG['config']['env_config'])
    obs = env.reset()
    while True:
        action = np.array([1])
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
    # CONFIG['config']['env_config']['sim']['max_t'] = 5.
    agent = ppo.PPOTrainer(config=CONFIG['config'], env=EnvLine)
    agent.restore(checkpoint_path)

    env = EnvLine(CONFIG['config']['env_config'])
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
    register_env("line", lambda env_config: EnvLine(env_config))

    checkpoint_paths = train()
    evaluate(checkpoint_paths)
    ray.shutdown()

    # sim_data = fym.logging.load(
    #     Path(Path(checkpoint_paths[-1][0]).parent, 'sim_data.h5')
    # )
    plot(Path(Path(checkpoint_paths[-1][0]).parent, 'sim_data.h5'))


def plot(sim_data_path):
    tex_fonts = {
        "text.usetex": True,
        "font.family": "Times New Roman",
        "axes.grid": True,
    }
    parent_path = Path(sim_data_path).parent
    plt.rcParams.update(tex_fonts)
    sim_data = fym.logging.load(sim_data_path)

    time = sim_data['t']
    pos = sim_data['pos'].ravel()
    vel = sim_data['vel'].ravel()
    lyapunov = sim_data['lyapunov']
    u = sim_data['u']

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(time, pos)
    ax[0].set_ylabel('position [m]')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, vel)
    ax[1].set_ylabel('velocity [m/s]')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[2].plot(time, u)
    ax[2].set_ylabel(r'action [$m/s^2$]')
    fig.suptitle('State and Action')
    fig.supxlabel('Time [s]')
    fig.align_ylabels(ax)
    fig.tight_layout()
    fig.savefig(Path(parent_path, "state.pdf"))
    plt.close('all')

    fig, ax = plt.subplots(1,1)
    ax.plot(time, lyapunov)
    ax.set_ylabel('C')
    ax.set_xlabel('Time [s]')
    ax.set_title('Condition value')
    fig.tight_layout()
    fig.savefig(Path(parent_path, "condition.pdf"))
    plt.close('all')


if __name__ == "__main__":
    main()
    # debug()

    # logdir = './ray_results/PPOTrainer_2022-05-24_01-23-32/PPOTrainer_quadcopter_b00e0_00000_0_2022-05-24_01-23-32/checkpoint_010000/'
    # checkpoint_path = Path(logdir, 'checkpoint-10000')
    # # evaluate([[str(checkpoint_path), 0]])
    # sim_data_path = Path(logdir, 'sim_data.h5')
    # plot(sim_data_path)


    


