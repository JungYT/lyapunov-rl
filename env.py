import numpy as np
import gym
from gym import spaces
from fym.core import BaseEnv
from scipy.spatial.transform import Rotation as rot
from numpy.linalg import norm

from plant import Multicopter, Line


class EnvMulticopter(BaseEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__(**env_config["sim"])
        self.plant = Multicopter(**env_config["init"])

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(15,))
        self.action_space = spaces.Box(
            low=np.float32(self.plant.rotorf_min),
            high=np.float32(self.plant.rotorf_max),
            shape=(self.plant.nrotors,),
        )
        angle_lim = 30
        self.state_space = spaces.Box(
            low = np.float32(np.hstack(
                [
                    [-10, -10, -10],
                    [-20, -20, -20],
                    np.deg2rad([-angle_lim, -angle_lim, -360]),
                    # [np.cos(angle_lim), -np.sin(angle_lim), 
                    #  np.cos(angle_lim), -np.sin(angle_lim), 
                    #  -1, -1],
                    [-20, -20, -20],
                ]
            )),
            high = np.float32(np.hstack(
                [
                    [10, 10, 10],
                    [20, 20, 20],
                    # [1, np.sin(angle_lim), 
                    #  1, np.sin(angle_lim), 
                    #  1, 1],
                    np.deg2rad([angle_lim, angle_lim, 360]),
                    [20, 20, 20],
                ]
            )),
        )

        self.P = np.diag([10, 10, 10, 1])
        self.Q = np.diag([
            10, 10, 10, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1
        ])
        self.R = np.diag([1, 1, 1, 1])

    def step(self, action):
        obs = self.observe()
        *_, done = self.update(action=action)
        next_obs = self.observe()
        reward = self.get_reward(obs, next_obs, action)
        done = done or not self.check_bound()
        return next_obs, reward, done, {}

    def set_dot(self, t, action):
        rotorfs_cmd = np.float64(action[:, None])
        rotorfs = self.plant.set_dot(t, rotorfs_cmd)
        obs = self.observe()
        V = self.lyapunov(obs)
        return dict(t=t, **self.plant.observe_dict(), **rotorfs, lyapunov=V,
                    obs=obs, rotorfs_cmd=rotorfs_cmd)

    def observe(self):
        pos, vel, R, omega = self.plant.observe_list()
        euler = rot.from_matrix(R).as_euler("ZYX")[::-1]
        attitude = np.array([
            np.cos(euler[0]), np.sin(euler[0]),
            np.cos(euler[1]), np.sin(euler[1]),
            np.cos(euler[2]), np.sin(euler[2])
        ])
        obs = np.hstack((pos.ravel(), vel.ravel(), attitude, omega.ravel()))
        return np.float32(obs)

    def get_reward(self, obs, next_obs, action):
        # reward = self.linear_comb(obs, next_obs, action)
        reward = self.lyapunov_guided(obs, next_obs, action)
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

    def linear_comb(self, obs, next_obs, action):
        if self.check_bound():
            pos = obs[0:3]
            vel = obs[3:6]
            omega = obs[12:]
            reward = -4e-3 * norm(pos) -5e-4 * norm(vel) - 3e-4 * norm(omega) \
                - 2e-4 * norm(action)
        else:
            t_current = self.clock.get()
            reward = -10 * (self.clock.max_t - t_current) / t_current
        return reward

    def lyapunov_guided(self, obs, next_obs, action):
        if self.check_bound():
            V = self.lyapunov(obs)
            V_next = self.lyapunov(next_obs)
            del_V = V_next - V
            exp = np.exp(-obs.T @ self.Q @ obs - action.T @ self.R @ action)
            if (del_V<=-1e-7 and V_next>1e-6) or (del_V<=0  and V_next<=1e-6):
                reward = -1 + exp
            else:
                reward = -10 + exp
        else:
            t_current = self.clock.get()
            reward = -10 * (self.clock.max_t - t_current) / t_current
        return reward

    def lyapunov(self, obs):
        pos = obs[0:3]
        omega = obs[12:]
        x = np.hstack((pos, omega[2]))
        V = x.T @ self.P @ x
        return V

    def check_bound(self):
        pos, vel, R, omega = self.plant.observe_list()
        euler = rot.from_matrix(R).as_euler("ZYX")[::-1]
        state = np.hstack((pos.ravel(), vel.ravel(), euler, omega.ravel()))
        return self.state_space.contains(state)
        

class EnvLine(BaseEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__(**env_config["sim"])
        self.plant = Line(**env_config["init"])

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,))
        self.action_space = spaces.Box(
            low=np.float32(self.plant.u_min),
            high=np.float32(self.plant.u_max),
            shape=(1,),
        )
        self.state_space = spaces.Box(
            low=np.float32(np.hstack([-10, -5])),
            high=np.float32(np.hstack([10, 5])),
        )
        # self.P = np.diag([10, 1])
        self.P = np.diag([1])
        self.Q = np.diag([10, 1])
        self.R = np.diag([1])

    def step(self, action):
        obs = self.observe()
        *_, done = self.update(action=action)
        next_obs = self.observe()
        reward = self.get_reward(obs, next_obs, action)
        return next_obs, reward, done, {}

    def set_dot(self, t, action):
        u = np.float64(action[:, None])
        self.plant.set_dot(t, u)
        obs = self.observe()
        V = self.lyapunov(obs)
        return dict(t=t, **self.plant.observe_dict(), lyapunov=V, u=action)

    def observe(self):
        pos, vel = self.plant.observe_list()
        obs = np.hstack((pos.ravel(), vel.ravel()))
        return np.float32(obs)

    def get_reward(self, obs, next_obs, action):
        V = self.lyapunov(obs)
        V_next = self.lyapunov(next_obs)
        del_V = V_next - V
        exp = np.exp(-obs.T @ self.Q @ obs - action.T @ self.R @ action)
        if (del_V<=-1e-7 and V_next>1e-6) or (del_V<=0  and V_next<=1e-6):
            reward = -1 + exp
        else:
            reward = -10 + exp
        return np.float32(reward)
        
    def reset(self, random=True):
        super().reset()
        if random:
            sample = np.float64(self.state_space.sample())
            self.plant.pos.state = np.vstack([sample[0]])
            self.plant.vel.state = np.vstack([sample[1]])
        return self.observe()

    def lyapunov(self, obs):
        pos, vel = obs
        # x = np.hstack((pos, vel))
        x = pos
        V = x.T @ self.P @ x
        return V
