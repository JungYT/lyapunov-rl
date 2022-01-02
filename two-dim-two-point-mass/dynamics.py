import numpy as np
import numpy.random as random

import gym
import fym
from fym.core import BaseEnv, BaseSystem


class TwoDimPointMass(BaseEnv):
    def __init__(self):
        super().__init__()
        self.pos = BaseSystem(shape=(2,1))
        self.vel = BaseSystem(shape=(2,1))

    def set_dot(self, u, disturbance):
        self.pos.dot = self.vel.state
        self.vel.dot = u + disturbance


class EnvTwoDimPointMass(BaseEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__(**env_config)
        self.plant = TwoDimPointMass()

        self.action_space = gym.spaces.Box(low=-10., high=10., shape=(2,))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.plant.state.shape
        )

    def reset(self, initial="random"):
        if initial == "random":
            self.plant.initial_state = 5 * (
                2*random.rand(*self.plant.state.shape) - 1
            )
        else:
            self.plant.initial_state = initial
        super().reset()
        obs = self.observe()
        return obs

    def step(self, action):
        pre_obs = self.observe()
        u = np.vstack(action)
        *_, done = self.update(u=u)
        next_obs = self.observe()
        reward = self.get_reward(pre_obs, next_obs, u)
        info = {}
        return next_obs, reward, done, info

    def set_dot(self, t, u):
        disturbance = random.normal(0, 0.1, (2,1))
        # disturbance = np.vstack((0., 0.))
        self.plant.set_dot(u, disturbance)
        x = np.float32(self.plant.state)
        pos = x[0:2]
        vel = x[2:4]
        lyap_dot = pos.squeeze() @ vel.squeeze()
        return dict(t=t, **self.observe_dict(), action=u, lyap_dot=lyap_dot)

    def observe(self):
        obs = np.float32(self.plant.state)
        return obs

    def get_reward(self, pre_obs, next_obs, u):
        # reward = self.L2norm()
        # reward = self.quadratic(u)
        # reward = self.exponential_quadratic()
        reward = self.lyapunov(pre_obs, next_obs)
        # reward = self.exponential_lyapunov()

        return reward

    def L2norm(self):
        x = np.float32(self.plant.state)
        pos = x[0:2]
        vel = x[2:4]
        reward = -5e-3 * np.linalg.norm(pos).item() \
            - 1e-5 * np.linalg.norm(vel).item()
        return reward

    def quadratic(self, u):
        x = np.float32(self.plant.state)
        reward = np.float32(
            (-x.T@np.diag([1, 1, 0, 0])@x 
             - u.T@np.diag([0, 0])@u).item()
        )
        return reward

    def exponential_quadratic(self):
        x = np.float32(self.plant.state)
        reward = np.float32(
            np.exp(
                1e-1 * (
                    -x.T @ np.diag([100, 100, 1, 1]) @ x 
                    - u.T @ np.diag([10, 10]) @ u
                ).item()
            )
        )
        return reward

    def lyapunov(self, pre_obs, next_obs):
        del_lyap = next_obs[0:2].squeeze() @ next_obs[0:2].squeeze() \
            - pre_obs[0:2].squeeze() @ pre_obs[0:2].squeeze()
        if del_lyap <= 0:
            reward = -1
        else:
            reward = -10
        # x = np.float32(self.plant.state)
        # pos = x[0:2]
        # vel = x[2:4]
        # lyap_dot = pos.squeeze() @ vel.squeeze()
        # if lyap_dot <= 0:
        #     reward = -1
        # else:
        #     reward = -10
        return reward

    def exponential_lyapunov(self):
        x = np.float32(self.plant.state)
        pos = x[0:2]
        vel = x[2:4]
        lyap_dot = pos.squeeze() @ vel.squeeze()
        if lyap_dot <= 0:
            reward = -3
        else:
            reward = -6
        tmp = np.float32(
            np.exp(
                1e-1 * (
                    -x.T @ np.diag([100, 100, 1, 1]) @ x 
                    - u.T @ np.diag([10, 10]) @ u
                ).item()
            )
        )
        reward += tmp
        return reward




