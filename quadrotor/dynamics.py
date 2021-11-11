import numpy as np
import numpy.random as random

import gym
from fym.core import BaseEnv, BaseSystem

def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])

class Quadrotor(BaseEnv):
    def __init__(self):
        super().__init__()
        self.pos = BaseSystem(shape=(3,1))
        self.vel = BaseSystem(shape=(3,1))
        self.C = BaseSystem(shape=(3,3))
        self.omega = BaseSystem(shape=(3,1))

        self.g = np.vstack((0, 0, -9.81))
        self.m = 4.34
        self.J = np.diag([0.0820, 0.0845, 0.1377])
        self.J_inv = np.linalg.inv(self.J)
        d = 0.315 # Distance from center of mass to center of each rotor
        ctf = 8.004e-4 # Torque coefficient. ``torquad_i = (-1)^i cft f_i``
        self.B = np.array([
            [1, 1, 1, 1],
            [0, -d, 0, d],
            [d, 0, -d, 0],
            [-ctf, ctf, -ctf, ctf]
        ])
        self.B_inv = np.linalg.inv(self.B)

    def set_dot(self, F, M):
        _, vel, C, omega = self.observe_list()
        omega_hat = hat(omega)
        self.pos.dot = vel
        self.vel.dot = self.g + C.T.dot(F)
        self.C.dot = -omega_hat.dot(C)
        self.omega.dot = self.J_inv.dot(M - omega_hat.dot(self.J.dot(omega)))

    def convert_thrust2FM(self, thrust):
        """Convert thust of each rotor to force and moment
        Parameters:
            thrust: (4,) list
        """
        return (self.B.dot(thrust)).ravel().tolist()

    def convert_FM2thrust(self, F, M):
        M1, M2, M3 = M
        return self.B_inv.dot(np.vstack((F, M1, M2, M3))).tolist()


class EnvQuadrotor(BaseEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__(**env_config)
        self.quad = Quadrotor()

        self.action_space = gym.spaces.Box(low=


        

