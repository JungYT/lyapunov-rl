import numpy as np
import gym
from gym import spaces
from fym.core import BaseEnv
from scipy.spatial.transform import Rotation as rot
from numpy.linalg import norm

from plant import Multicopter, Line, ThreeDOF, SecondOrder


class EnvQuadHovering(BaseEnv, gym.Env):
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
                    np.deg2rad([-angle_lim, -angle_lim, -180]),
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
                    np.deg2rad([angle_lim, angle_lim, 180]),
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
        done = done or not self.check_contain()
        info = {'x': self.check_contain(), 't': self.clock.get()}
        return next_obs, reward, done, info

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
        if self.check_contain():
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
        if self.check_contain():
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

    def check_contain(self):
        pos, vel, R, omega = self.plant.observe_list()
        euler = rot.from_matrix(R).as_euler("ZYX")[::-1]
        state = np.hstack((pos.ravel(), vel.ravel(), euler, omega.ravel()))
        return self.state_space.contains(np.float32(state))
        

class EnvQuadAttitude(BaseEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__(**env_config["sim"])
        self.plant = Multicopter(**env_config["init"])

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,))
        self.action_space = spaces.Box(
            low=np.float32(self.plant.rotorf_min),
            high=np.float32(self.plant.rotorf_max),
            shape=(self.plant.nrotors,),
        )
        angle_lim = 30
        self.state_space = spaces.Box(
            low = np.float32(np.hstack(
                [
                    np.deg2rad([-angle_lim, -angle_lim, -180]),
                    [-20, -20, -20],
                ]
            )),
            high = np.float32(np.hstack(
                [
                    np.deg2rad([angle_lim, angle_lim, 180]),
                    [20, 20, 20],
                ]
            )),
        )

        self.P = np.diag([10, 10, 10, 10, 1, 1, 0.1])
        self.Q = np.diag([
            10, 10, 10, 10, 1, 1, 0.1, 0.1, 0.1
        ])
        self.R = np.diag([1, 1, 1, 1])

    def step(self, action):
        obs = self.observe()
        *_, done = self.update(action=action)
        next_obs = self.observe()
        reward = self.get_reward(obs, next_obs, action)
        done = done or not self.check_contain()
        # info = {'x': self.check_contain(), 't': self.clock.get()}
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
            np.cos(euler[0]) - 1, np.sin(euler[0]),
            np.cos(euler[1]) - 1, np.sin(euler[1]),
            np.cos(euler[2]) - 1, np.sin(euler[2])
        ])
        obs = np.hstack((attitude, omega.ravel()))
        return np.float32(obs)

    def get_reward(self, obs, next_obs, action):
        # reward = self.linear_comb(obs, next_obs, action)
        reward = self.lyapunov_guided(obs, next_obs, action)
        return np.float32(reward)

    def reset(self, random=True):
        super().reset()
        if random:
            sample = np.float64(self.state_space.sample())
            self.plant.pos.state = np.zeros((3, 1))
            self.plant.vel.state = np.zeros((3, 1))
            self.plant.R.state = rot.from_euler(
                "ZYX", 
                sample[:3][::-1]
            ).as_matrix()
            self.plant.omega.state = sample[3:][:, None]
        return self.observe()

    def linear_comb(self, obs, next_obs, action):
        if self.check_contain():
            att = obs[:6]
            omega = obs[6:]
            reward = -1e-2 * norm(att) - 3e-4 * norm(omega) \
                - 2e-4 * norm(action)
        else:
            t_current = self.clock.get()
            reward = -10 * (self.clock.max_t - t_current) / t_current
        return reward

    def lyapunov_guided(self, obs, next_obs, action):
        if self.check_contain():
            V = self.lyapunov(obs)
            V_next = self.lyapunov(next_obs)
            del_V = V_next - V
            exp = np.exp(-obs.T @ self.Q @ obs - action.T @ self.R @ action)
            if (del_V<=-1e-6 and V_next>1e-5) or (del_V<=0  and V_next<=1e-5):
                reward = -1 + exp
            else:
                reward = -10 + exp
        else:
            t_current = self.clock.get()
            reward = -10 * (self.clock.max_t - t_current) / t_current
        return reward

    def lyapunov(self, obs):
        att = obs[:6]
        omega = obs[6:]
        x = np.hstack((att, omega[2]))
        V = x.T @ self.P @ x
        return V

    def check_contain(self):
        pos, vel, R, omega = self.plant.observe_list()
        euler = rot.from_matrix(R).as_euler("ZYX")[::-1]
        state = np.hstack((euler, omega.ravel()))
        return self.state_space.contains(np.float32(state))


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


class EnvStandOff(BaseEnv, gym.Env):
    r_des = 1
    def __init__(self, env_config):
        super().__init__(**env_config['sim'])
        self.plant = ThreeDOF(**env_config['init'])

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,))
        self.action_space = spaces.Box(low=-3, high=3, shape=(1,))
        self.state_space = spaces.Box(
            low=np.float32(np.array([-5, -5, -np.pi])),
            high=np.float32(np.array([5, 5, np.pi])),
        )
        self.R = np.vstack([1])

    def step(self, action):
        obs = self.observe()
        *_, done = self.update(action=action)
        next_obs = self.observe()
        reward = self.get_reward(obs, next_obs, action)
        # done = done or not self.check_contain()
        # info = {'x': self.check_contain(), 't': self.clock.get()}
        return next_obs, reward, done, {}

    def set_dot(self, t, action):
        u = np.float64(action[:, None])
        self.plant.set_dot(t, u)
        return dict(t=t, **self.plant.observe_dict(), action=action)

    def observe(self):
        pos, yaw = self.plant.observe_list()
        r = norm(pos)
        e_r = r - self.r_des
        sin_theta = pos[1] / r
        cos_theta = pos[0] / r
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)
        obs = np.hstack((
            e_r, cos_theta, sin_theta, cos_yaw.ravel(), sin_yaw.ravel()
        ))
        return np.float32(obs)

    def get_reward(self, obs, next_obs, action):
        # reward = self.linear_comb(obs, next_obs, action)
        reward = self.lyapunov_guided(obs, next_obs, action)
        return np.float32(reward)

    def reset(self, random=True):
        super().reset()
        if random:
            sample = np.float64(self.state_space.sample())
            self.plant.pos.state = sample[:2][:, None]
            self.plant.yaw.state = np.vstack([sample[2]])
        return self.observe()

    def linear_comb(self, obs, next_obs, action):
        e_r = obs[0]
        reward = -e_r**2 - 5e-2*action**2
        return reward

    def lyapunov_guided(self, obs, next_obs, action):
        V = self.lyapunov(obs)
        V_next = self.lyapunov(next_obs)
        del_V = V_next - V
        exp = np.exp(- action.T @ self.R @ action)
        if (del_V<=-1e-6 and V_next>1e-5) or (del_V<=0  and V_next<=1e-5):
            reward = 10 + 5*exp
        else:
            reward = 1 + exp
        return reward

    def lyapunov(self, obs):
        e_r = obs[0]
        V = e_r**2
        return V

    def check_contain(self):
        pos, yaw = self.plant.observe_list()
        state = np.hstack((pos.ravel(), yaw.ravel()))
        return self.state_space.contains(np.float32(state))
        

class EnvStandOffSecondOrder(BaseEnv, gym.Env):
    r_des = 1
    def __init__(self, env_config):
        super().__init__(**env_config['sim'])
        self.plant = ThreeDOF(**env_config['init'])
        self.second_order = SecondOrder(
            u_min=self.plant.u_min, u_max=self.plant.u_max
        )

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,))
        self.action_space = spaces.Box(low=-3, high=3, shape=(1,))
        self.state_space = spaces.Box(
            low=np.float32(np.array([-5, -5, -np.pi])),
            high=np.float32(np.array([5, 5, np.pi])),
        )
        self.R = np.vstack([0.1])

    def step(self, action):
        obs = self.observe()
        *_, done = self.update(action=action)
        next_obs = self.observe()
        reward = self.get_reward(obs, next_obs, action)
        # done = done or not self.check_contain()
        # info = {'x': self.check_contain(), 't': self.clock.get()}
        return next_obs, reward, done, {}

    def set_dot(self, t, action):
        acc_cmd = np.float64(action[:, None])
        self.second_order.set_dot(t, acc_cmd)
        acc, jerk = self.second_order.observe_list()
        self.plant.set_dot(t, acc)
        return dict(t=t, **self.plant.observe_dict(), acc_cmd=acc_cmd, acc=acc,
                    jerk=jerk)

    def observe(self):
        pos, yaw = self.plant.observe_list()
        r = norm(pos)
        e_r = r - self.r_des
        sin_theta = pos[1] / r
        cos_theta = pos[0] / r
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)
        obs = np.hstack((
            e_r, cos_theta, sin_theta, cos_yaw.ravel(), sin_yaw.ravel()
        ))
        return np.float32(obs)

    def get_reward(self, obs, next_obs, action):
        # reward = self.linear_comb(obs, next_obs, action)
        reward = self.lyapunov_guided(obs, next_obs, action)
        return np.float32(reward)

    def reset(self, random=True):
        super().reset()
        if random:
            sample = np.float64(self.state_space.sample())
            self.plant.pos.state = sample[:2][:, None]
            self.plant.yaw.state = np.vstack([sample[2]])
        return self.observe()

    def linear_comb(self, obs, next_obs, action):
        e_r = obs[0]
        reward = -e_r**2 - 5e-2*action**2
        return reward

    def lyapunov_guided(self, obs, next_obs, action):
        V = self.lyapunov(obs)
        V_next = self.lyapunov(next_obs)
        del_V = V_next - V
        exp = np.exp(- action.T @ self.R @ action)
        if (del_V<=-1e-6 and V_next>1e-5) or (del_V<=0  and V_next<=1e-5):
            reward = -1 + exp
        else:
            reward = -10 + exp
        return reward

    def lyapunov(self, obs):
        e_r = obs[0]
        V = e_r**2
        return V

    def check_contain(self):
        pos, yaw = self.plant.observe_list()
        state = np.hstack((pos.ravel(), yaw.ravel()))
        return self.state_space.contains(np.float32(state))


class EnvQuad(BaseEnv):
    def __init__(self, env_config):
        super().__init__(**env_config["sim"])
        self.plant = Multicopter(**env_config['quad']['init'])

    def step(self, guidance_cmd):
        rotorfs_cmd = self.controller(guidance_cmd)
        *_, done = self.update(action=rotorfs_cmd)
        next_obs = self.observe()
        return next_obs, {}, done, {}

    def set_dot(self, t, action):
        rotorfs = self.plant.set_dot(t, action)
        obs = self.observe()
        return dict(t=t, **self.plant.observe_dict(), obs=obs, rotorfs=rotorfs,
                    rotorfs_cmd=action)

    def controller(self, guidance_cmd):


        return np.float64(rotorfs_cmd)

    def observer(self):
        pos, vel, R, omega = self.plant.observe_list()
        euler = rot.from_matrix(R).as_euler("ZYX")[::-1]
        obs = np.hstack((pos.ravel(), vel.ravel(), euler, omega.ravel()))
        return np.float32(obs)

    def reset(self):
        super().reset()
        return self.observe()


