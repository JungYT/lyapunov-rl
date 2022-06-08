import numpy as np
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from scipy.spatial.transform import Rotation as rot
from pathlib import Path
from matplotlib import pyplot as plt

import fym
from env import EnvStandOff, EnvStandOffSecondOrder

CONFIG = {
    "config": {
        "env": "standoff",
        "env_config": {
            "sim": {
                "dt": 0.01,
                "max_t": 10.,
            },
            "init": {
                "pos": np.vstack([3, 3]),
                "yaw": np.vstack([0]),
            },
        },
        "no_done_at_end": True,
        # "framework": "torch",
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
    "local_dir": "./ray_results/standoff",
    "checkpoint_freq": 200,
    "checkpoint_at_end": True,
}

def debug():
    env = EnvStandOff(CONFIG['config']['env_config'])
    obs = env.reset()
    while True:
        action = np.array([1])
        obs, reward, done, _ = env.step(action)
        # print(info['x'])
        # print(info['t'])
        # print(done)
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
    CONFIG['config']['env_config']['sim']['max_t'] = 15.
    agent = ppo.PPOTrainer(config=CONFIG['config'], env=EnvStandOff)
    agent.restore(checkpoint_path)

    env = EnvStandOffSecondOrder(CONFIG['config']['env_config'])
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
    register_env("standoff", lambda env_config: EnvStandOff(env_config))

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
    pos = sim_data['pos']
    yaw = sim_data['yaw'].ravel() * 180/np.pi
    # u = sim_data['action']
    acc_cmd = sim_data['acc_cmd']
    acc = sim_data['acc'].ravel()
    jerk = sim_data['jerk'].ravel()
    # rotorfs_cmd = sim_data['rotorfs_cmd']

    refx = np.cos(time * 2 * np.pi / time[-1])
    refy = np.sin(time * 2 * np.pi / time[-1])
    fig, ax = plt.subplots(1, 1)
    ax.plot(pos[:,0], pos[:,1], 'r', label='Traj.')
    ax.plot(refx, refy, 'b--', label='Ref.')
    ax.set_ylabel('Y [m]')
    ax.set_xlabel('X [m]')
    ax.set_title('Trajectory')
    plt.legend()
    fig.savefig(Path(parent_path, "trajectory.pdf"), bbox_inches='tight')
    plt.close('all')


    fig, ax = plt.subplots(3, 1)
    ax[0].plot(time, pos[:, 0])
    ax[0].set_ylabel('X [m]')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, pos[:, 1])
    ax[1].set_ylabel('Y [m]')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[2].plot(time, yaw)
    ax[2].set_ylabel(r'$\psi$ [deg]')
    fig.suptitle('Position and Yaw Angle')
    fig.supxlabel('Time [s]')
    fig.align_ylabels(ax)
    fig.tight_layout()
    fig.savefig(Path(parent_path, "state.pdf"))
    plt.close('all')

    # fig, ax = plt.subplots(1,1)
    # ax.plot(time, u)
    # ax.set_ylabel(r'Lateral acceleration [$m/s^2$]')
    # ax.set_xlabel('Time [s]')
    # ax.set_title('Guidance command')
    # fig.tight_layout()
    # fig.savefig(Path(parent_path, "acc.pdf"))
    # plt.close('all')

    fig, ax = plt.subplots(1,1)
    line1, = ax.plot(time, acc_cmd, 'b--')
    line2, = ax.plot(time, acc, 'r')
    ax.legend(handles=(line1, line2),
                    labels=('acc. cmd', 'acc.'))
    ax.set_ylabel(r'Lateral acceleration [$m/s^2$]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Acceleration Command and Acceleration')
    fig.tight_layout()
    fig.savefig(Path(parent_path, "acc.pdf"))
    plt.close('all')

if __name__ == "__main__":
    main()
    # debug()

    # logdir = './ray_results/standoff'
    # checkpoint_path = Path(logdir, 'checkpoint-2000')
    # evaluate([[str(checkpoint_path), 0]])
    # sim_data_path = Path(logdir, 'sim_data.h5')
    # plot(sim_data_path)


    


