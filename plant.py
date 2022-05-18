import numpy as np
from numpy.linalg import norm
from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import hat


def cross(x, y):
    return np.cross(x, y, axis=0)


class Multicopter(BaseEnv):
    g = 9.81
    m = 1.00
    r = 0.24
    J = np.diag([8.1, 8.1, 14.2]) * 1e-3
    Jinv = np.linalg.inv(J)
    b = 5.42e-5
    d = 1.1e-6
    Kf = np.diag([5.567, 5.567, 6.354]) * 1e-4
    Kt = np.diag([5.567, 5.567, 6.354]) * 1e-4
    rotorf_min = 0
    rotorf_max = 20
    e3 = np.vstack((0, 0, 1))
    nrotors = 4
    B = np.array([
        [1, 1, 1, 1],
        [0, -r, 0, r],
        [r, 0, -r, 0],
        [-d / b, d / b, -d / b, d / b],
    ])
    Lambda = np.eye(4)

    def __init__(self, pos, vel, R, omega):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.R = BaseSystem(R)
        self.omega = BaseSystem(omega)

    def deriv(self, pos, vel, R, omega, rotorfs):
        u = self.B @ rotorfs
        fT, M = u[:1], u[1:]

        dpos = vel
        dvel = (
            (self.m * self.g * self.e3 + R @ (-fT * self.e3) - self.Kf @ vel)
            / self.m
        )
        dR = R @ hat(omega)
        domega = self.Jinv @ (
            M - cross(omega, self.J @ omega) - norm(omega) * self.Kt @ omega
        )
        return dpos, dvel, dR, domega

    def set_dot(self, t, rotorfs_cmd):
        pos, vel, R, omega = self.observe_list()
        rotorfs = self.set_valid(t, rotorfs_cmd)
        dots = self.deriv(pos, vel, R, omega, rotorfs)
        self.pos.dot, self.vel.dot, self.R.dot, self.omega.dot = dots
        return dict(rotorfs=rotorfs)

    def set_valid(self, t, rotorfs_cmd):
        rotorfs = np.clip(rotorfs_cmd, self.rotorf_min, self.rotorf_max)
        return self.Lambda @ rotorfs

