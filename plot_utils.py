import numpy as np


def get_circle(x0, y0, z0, r, num=100):
    theta = np.linspace(0, 2 * np.pi, num)
    xs = x0 + r * np.cos(theta)
    ys = y0 + r * np.sin(theta)
    zs = z0 + np.zeros(100)
    return xs, ys, zs


def get_voxel(x0, y0, z0, length, width, height):
    dx = length / 2
    dy = width / 2
    dz = height / 2
    xs = np.array([-dx, dx, -dx, dx, -dx, dx, -dx, dx]).reshape(2, 2, 2) + x0
    ys = np.array([-dy, -dy, dy, dy, -dy, -dy, dy, dy]).reshape(2, 2, 2) + y0
    zs = np.array([-dz, -dz, -dz, -dz, dz, dz, dz, dz]).reshape(2, 2, 2) + z0
    filled = np.ones((1, 1, 1))
    return xs, ys, zs, filled
