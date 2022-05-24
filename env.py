import numpy as np
import gym
from gym import spaces
from fym.core import BaseEnv
from scipy.spatial.transform import Rotation as rot
from numpy.linalg import norm

from plant import Multicopter


class Env(BaseEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__(**env_config["sim"])
        self.plant = Multicopter(**env_config["init"])

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,))
        self.action_space = spaces.Box(
            low=np.float32(self.plant.rotorf_min),
            high=np.float32(self.plant.rotorf_max),
            shape=(self.plant.nrotors,),
        )
        self.state_space = spaces.Box(
            low = np.float32(np.hstack(
                [
                    [-10, -10, -10],
                    [-20, -20, -20],
                    np.deg2rad([-45, -45, -360]),
                    [-10, -10, -10],
                ]
            )),
            high = np.float32(np.hstack(
                [
                    [10, 10, 10],
                    [20, 20, 20],
                    np.deg2rad([45, 45, 360]),
                    [10, 10, 10],
                ]
            )),
        )

        self.P = np.diag([10, 10, 10, 10, 10, 10, 1])
        self.Q = np.diag([
            10, 10, 10, 10, 10, 10, 1
        ])
        self.R = np.diag([2, 2, 2, 2])

    def step(self, action):
        obs = self.observe()
        *_, done = self.update(action=action)
        next_obs = self.observe()
        reward = self.get_reward(obs, next_obs, action)
        # done = done or not self.state_space.contains(next_obs)
        return next_obs, reward, done, {}

    def set_dot(self, t, action):
        rotorfs_cmd = np.float64(action[:, None])
        rotorfs = self.plant.set_dot(t, rotorfs_cmd)
        obs = self.observe()
        V = self.lyapunov(obs)
        
        return dict(t=t, **self.plant.observe_dict(), **rotorfs, lyapunov=V,
                    obs=obs)

    def observe(self):
        pos, vel, R, omega = self.plant.observe_list()
        euler = rot.from_matrix(R).as_euler("ZYX")[::-1]
        attitude = np.array([
            np.cos(euler[0]) - 1, np.sin(euler[0]),
            np.cos(euler[1]) - 1, np.sin(euler[1]),
            np.cos(euler[2]) - 1, np.sin(euler[2])
        ])
        obs = np.hstack((attitude, omega.ravel()))
        return np.float32(obs)

    def get_reward(self, obs, next_obs, action):
        attitude = obs[0:6]
        omega = obs[6:9]
        reward = -5e-3 * norm(attitude) - 3e-4 * norm(omega) \
            - 2e-4 * norm(action)
        # V = self.lyapunov(obs)
        # V_next = self.lyapunov(next_obs)
        # del_V = V_next - V
        # exp = np.exp(-obs.T @ self.Q @ obs - action.T @ self.R @ action)
        # if (del_V<=-1e-7 and V_next>1e-6) or (del_V<=0  and V_next<=1e-6):
        #     reward = -1 + exp
        # else:
        #     reward = -10 + exp
        return np.float32(reward)

    def reset(self, random=True):
        super().reset()
        if random:
            sample = np.float64(self.state_space.sample())
            self.plant.pos.state = sample[:3][:, None]
            self.plant.vel.state = sample[3:6][:, None]
            self.plant.R.state = rot.from_euler(
                "ZYX", 
                sample[6:9][::-1]
            ).as_matrix()
            self.plant.omega.state = sample[9:12][:, None]
        return self.observe()

    def lyapunov(self, obs):
        # pos = obs[0:3]
        attitude = obs[0:6]
        omega = obs[6:9]
        x = np.hstack((attitude, omega[2]))
        V = x.T @ self.P @ x
        return V
        
