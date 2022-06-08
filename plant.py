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
        rotorfs = self.saturate(t, rotorfs_cmd)
        dots = self.deriv(pos, vel, R, omega, rotorfs)
        self.pos.dot, self.vel.dot, self.R.dot, self.omega.dot = dots
        return dict(rotorfs=rotorfs)

    def saturate(self, t, rotorfs_cmd):
        rotorfs = np.clip(rotorfs_cmd, self.rotorf_min, self.rotorf_max)
        return self.Lambda @ rotorfs


class Line(BaseEnv):
    u_min = -10.
    u_max = 10.
    def __init__(self, pos, vel):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)

    def deriv(self, pos, vel, u):
        dvel = u
        dpos = vel
        return dpos, dvel

    def set_dot(self, t, action):
        pos, vel = self.observe_list()
        u = self.saturate(action)
        dots = self.deriv(pos, vel, u)
        self.pos.dot, self.vel.dot = dots

    def saturate(self, action):
        u = np.clip(action, self.u_min, self.u_max)
        return u


class ThreeDOF(BaseEnv):
    V = 1
    u_min = -3
    u_max = 3
    def __init__(self, pos, yaw):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.yaw = BaseSystem(yaw)

    def set_dot(self, t, u):
        yaw = self.yaw.state.squeeze()
        self.pos.dot = np.vstack((
            self.V * np.cos(yaw),
            self.V * np.sin(yaw)
        ))
        self.yaw.dot = u / self.V


class SecondOrder(BaseEnv):
    freq = 100
    damp = 0.707
    def __init__(self, u_min=None, u_max=None):
        super().__init__()
        self.x = BaseSystem(np.vstack([0]))
        self.xdot = BaseSystem(np.vstack([0]))
        self.u_min = u_min
        self.u_max = u_max

    def deriv(self, x, xdot, u):
        dx = xdot
        dxdot = -self.freq**2 * x - 2 * self.damp * self.freq * xdot \
            + self.freq**2 * u
        return dx, dxdot

    def set_dot(self, t, u):
        x, xdot = self.observe_list()
        dots = self.deriv(x, xdot, u)
        self.x.dot, self.xdot.dot = dots

    def saturate(self, action):
        if self.u_min or self.u_max:
            u = np.clip(action, self.u_min, self.u_max)
        else:
            u = action
        return u
