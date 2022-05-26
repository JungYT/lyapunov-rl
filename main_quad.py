import numpy as np
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from scipy.spatial.transform import Rotation as rot
from pathlib import Path
from matplotlib import pyplot as plt

import fym
from env import EnvMulticopter

CONFIG = {
    "config": {
        "env": "quadcopter",
        "env_config": {
            "sim": {
                "dt": 0.01,
                "max_t": 5.,
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
        # "lr": 0.0001,
        # "gamma": 0.99,
        "lr": tune.grid_search([0.001, 0.0005, 0.0001]),
        "gamma": tune.grid_search([0.9, 0.99, 0.999]),
    },
    "stop": {
        "training_iteration": 2000,
    },
    "local_dir": "./ray_results/quadcopter",
    "checkpoint_freq": 100,
    "checkpoint_at_end": True,
}

def debug():
    env = EnvMulticopter(CONFIG['config']['env_config'])
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
    CONFIG['config']['env_config']['sim']['max_t'] = 10.
    agent = ppo.PPOTrainer(config=CONFIG['config'], env=EnvMulticopter)
    agent.restore(checkpoint_path)

    env = EnvMulticopter(CONFIG['config']['env_config'])
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
    register_env("quadcopter", lambda env_config: EnvMulticopter(env_config))

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
    R = sim_data['R']
    euler = rot.from_matrix(R).as_euler("ZYX")[::-1] * 180/np.pi
    omega = sim_data['omega']
    rotorfs = sim_data['rotorfs']
    # rotorfs_cmd = sim_data['rotorfs_cmd']

    fig, ax = plt.subplots(3, 2)
    ax[0][0].plot(time, euler[:,0])
    ax[0][0].set_ylabel(r'$\phi$ [deg]')
    ax[0][0].axes.xaxis.set_ticklabels([])
    ax[0][1].plot(time, omega[:,0])
    ax[0][1].set_ylabel(r'$\omega_x$ [deg/s]')
    ax[0][1].axes.xaxis.set_ticklabels([])
    ax[1][0].plot(time, euler[:,1])
    ax[1][0].set_ylabel(r'$\theta$ [deg]')
    ax[1][0].axes.xaxis.set_ticklabels([])
    ax[1][1].plot(time, omega[:,1])
    ax[1][1].set_ylabel(r'$\omega_y$ [deg/s]')
    ax[1][1].axes.xaxis.set_ticklabels([])
    ax[2][0].plot(time, euler[:,2])
    ax[2][0].set_ylabel(r'$\psi$ [deg]')
    ax[2][1].plot(time, omega[:,2])
    ax[2][1].set_ylabel(r'$\omega_z$ [deg/s]')
    fig.suptitle('Euler Angle and Angular Rate')
    fig.supxlabel('Time [s]')
    fig.align_ylabels(ax)
    fig.tight_layout()
    fig.savefig(Path(parent_path, "attitude.pdf"))
    plt.close('all')

    fig, ax = plt.subplots(2, 2)
    # line1, = ax[0][0].plot(time, rotorfs[:, 0], 'r')
    # line2, = ax[0][0].plot(time, rotorfs_cmd[:, 0], 'b--')
    # ax[0][0].legend(handles=(line1, line2),
    #                 labels=('rotor force', 'rotor force command'))
    ax[0][0].plot(time, rotorfs[:,0])
    ax[0][0].set_title('rotor 1')
    ax[0][0].axes.xaxis.set_ticklabels([])
    # ax[0][1].plot(time, rotorfs[:,1], 'r', time, rotorfs_cmd[:,1], 'b--')
    ax[0][1].plot(time, rotorfs[:,1])
    ax[0][1].set_title('rotor 2')
    ax[0][1].axes.xaxis.set_ticklabels([])
    # ax[1][0].plot(time, rotorfs[:,1], 'r', time, rotorfs_cmd[:,1], 'b--')
    ax[1][0].plot(time, rotorfs[:,1])
    ax[1][0].set_title('rotor 3')
    # ax[1][1].plot(time, rotorfs[:,1], 'r', time, rotorfs_cmd[:,1], 'b--')
    ax[1][1].plot(time, rotorfs[:,1])
    ax[1][1].set_title('rotor 4')
    fig.supylabel('Rotor force [N]')
    fig.supxlabel('Time [s]')
    fig.suptitle('Rotor force')
    fig.tight_layout()
    fig.savefig(Path(parent_path, "rotor.pdf"))
    plt.close('all')


if __name__ == "__main__":
    main()
    # debug()

    # logdir = './ray_results/quadcopter/PPOTrainer_2022-05-24_17-07-26/PPOTrainer_quadcopter_8ca97_00000_0_2022-05-24_17-07-26/checkpoint_010000/'
    # checkpoint_path = Path(logdir, 'checkpoint-10000')
    # # evaluate([[str(checkpoint_path), 0]])
    # sim_data_path = Path(logdir, 'sim_data.h5')
    # plot(sim_data_path)


    


