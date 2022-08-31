import numpy as np


class KalmanFilter2D:
    def __init__(self, dt, x, vx, sigma_ax=1, sigma_ox=1):
        self.dt = dt
        gg = np.array([[0.5 * self.dt ** 2],
                       [self.dt]])
        self.noise_q = gg @ np.array([[sigma_ax ** 2]]) @ gg.T
        self.noise_r = np.array([[sigma_ox ** 2]])
        self.xx = np.array([[x],
                            [vx]])
        self.pp = np.array([[sigma_ox ** 2, 0],
                            [0, 0]])
        self.ff = np.array([[1, self.dt],
                            [0, 1]])
        self.hh = np.array([[1, 0]])

    def predict(self):
        self.xx = self.ff @ self.xx
        self.pp = self.ff @ self.pp @ self.ff.T + self.noise_q

    def update(self, zx):
        zs = np.array([[zx]])
        zz = zs - self.hh @ self.xx
        ss = self.hh @ self.pp @ self.hh.T + self.noise_r
        kk = self.pp @ self.hh.T @ np.linalg.inv(ss)
        self.xx = self.xx + kk @ zz
        self.pp = self.pp - kk @ self.hh @ self.pp

    def get_location(self):
        return self.xx[0, 0]

    def get_velocity(self):
        return self.xx[1, 0]

    def get_state(self):
        return self.xx[0, 0], self.xx[1, 0]

    def set_state(self, x, vx):
        self.xx[0, 0] = x
        self.xx[1, 0] = vx


class KalmanFilter4D:
    def __init__(self, dt, x, vx, y, vy, sigma_ax=1, sigma_ay=1, sigma_ox=1, sigma_oy=1):
        self.dt = dt
        gg = np.array([[0.5 * self.dt ** 2, 0],
                       [self.dt, 0],
                       [0, 0.5 * self.dt ** 2],
                       [0, self.dt]])
        self.noise_q = gg @ np.array([[sigma_ax ** 2, 0], [0, sigma_ay ** 2]]) @ gg.T
        self.noise_r = np.array([[sigma_ox ** 2, 0], [0, sigma_oy ** 2]])
        self.xx = np.array([[x],
                            [vx],
                            [y],
                            [vy]])
        self.pp = np.array([[sigma_ox ** 2, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, sigma_oy ** 2, 0],
                            [0, 0, 0, 0]])
        self.ff = np.array([[1, self.dt, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, self.dt],
                            [0, 0, 0, 1]])
        self.hh = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0]])

    def predict(self):
        self.xx = self.ff @ self.xx
        self.pp = self.ff @ self.pp @ self.ff.T + self.noise_q

    def update(self, zx, zy):
        zs = np.array([[zx],
                       [zy]])
        zz = zs - self.hh @ self.xx
        ss = self.hh @ self.pp @ self.hh.T + self.noise_r
        kk = self.pp @ self.hh.T @ np.linalg.inv(ss)
        self.xx = self.xx + kk @ zz
        self.pp = self.pp - kk @ self.hh @ self.pp

    def get_location(self):
        return self.xx[0, 0], self.xx[2, 0]

    def get_velocity(self):
        return self.xx[1, 0], self.xx[3, 0]

    def get_state(self):
        return self.xx[0, 0], self.xx[1, 0], self.xx[2, 0], self.xx[3, 0]

    def set_state(self, x, vx, y, vy):
        self.xx[0, 0] = x
        self.xx[1, 0] = vx
        self.xx[2, 0] = y
        self.xx[3, 0] = vy


class KalmanFilter6D:
    def __init__(self, dt, x, vx, y, vy, z, vz, sigma_ax=1, sigma_ay=1, sigma_az=1, sigma_ox=1, sigma_oy=1, sigma_oz=1):
        self.dt = dt
        gg = np.array([[0.5 * self.dt ** 2, 0, 0],
                       [self.dt, 0, 0],
                       [0, 0.5 * self.dt ** 2, 0],
                       [0, self.dt, 0],
                       [0, 0, 0.5 * self.dt ** 2],
                       [0, 0, self.dt]])
        self.noise_q = gg @ np.array([[sigma_ax ** 2, 0, 0], [0, sigma_ay ** 2, 0], [0, 0, sigma_az ** 2]]) @ gg.T
        self.noise_r = np.array([[sigma_ox ** 2, 0, 0], [0, sigma_oy ** 2, 0], [0, 0, sigma_oz ** 2]])
        self.xx = np.array([[x],
                            [vx],
                            [y],
                            [vy],
                            [z],
                            [vz]])
        self.pp = np.array([[sigma_ox ** 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, sigma_oy ** 2, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, sigma_oz ** 2, 0],
                            [0, 0, 0, 0, 0, 0]])
        self.ff = np.array([[1, self.dt, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, self.dt, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, self.dt],
                            [0, 0, 0, 0, 0, 1]])
        self.hh = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0]])

    def predict(self):
        self.xx = self.ff @ self.xx
        self.pp = self.ff @ self.pp @ self.ff.T + self.noise_q

    def update(self, zx, zy, zz):
        zs = np.array([[zx],
                       [zy],
                       [zz]])
        zz = zs - self.hh @ self.xx
        ss = self.hh @ self.pp @ self.hh.T + self.noise_r
        kk = self.pp @ self.hh.T @ np.linalg.inv(ss)
        self.xx = self.xx + kk @ zz
        self.pp = self.pp - kk @ self.hh @ self.pp

    def get_location(self):
        return self.xx[0, 0], self.xx[2, 0], self.xx[4, 0]

    def get_velocity(self):
        return self.xx[1, 0], self.xx[3, 0], self.xx[5, 0]

    def get_state(self):
        return self.xx[0, 0], self.xx[1, 0], self.xx[2, 0], self.xx[3, 0], self.xx[4, 0], self.xx[5, 0]

    def set_state(self, x, vx, y, vy, z, vz):
        self.xx[0, 0] = x
        self.xx[1, 0] = vx
        self.xx[2, 0] = y
        self.xx[3, 0] = vy
        self.xx[4, 0] = z
        self.xx[5, 0] = vz
