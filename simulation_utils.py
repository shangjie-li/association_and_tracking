import numpy as np


def init_targets(target_num, iter_num):
    targets = np.zeros((9 * target_num, iter_num + 1))
    for t in range(target_num):
        targets[9 * t + 0, 0] = 1500 * np.random.rand() + 100  # x location
        targets[9 * t + 1, 0] = 5 * np.random.randn() - 20  # x velocity
        targets[9 * t + 2, 0] = 40 * np.random.randn()  # y location
        targets[9 * t + 3, 0] = 1 * np.random.randn()  # y velocity
        targets[9 * t + 4, 0] = 0  # z location
        targets[9 * t + 5, 0] = 0  # z velocity
        targets[9 * t + 6, 0] = 6.0  # length
        targets[9 * t + 7, 0] = 6.0  # width
        targets[9 * t + 8, 0] = 6.0  # height
    return targets


def control_target_state(state, dt, sigma_ax, sigma_ay, sigma_az):
    state = state.reshape(6, 1)
    v = np.array([sigma_ax * np.random.randn(), sigma_ay * np.random.randn(), sigma_az * np.random.randn()])
    v = v.reshape(3, 1)
    f = np.array([[1, dt, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, dt, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, dt],
                  [0, 0, 0, 0, 0, 1],
                  ])
    g = np.array([[0.5 * dt ** 2, 0, 0],
                  [dt, 0, 0],
                  [0, 0.5 * dt ** 2, 0],
                  [0, dt, 0],
                  [0, 0, 0.5 * dt ** 2],
                  [0, 0, dt],
                  ])
    return f @ state + g @ v  # [6, 1]


def control_target_shape(shape, sigma_vl, sigma_vw, sigma_vh, min_size=4.0, max_size=10.0):
    shape = shape.reshape(3, 1)
    s = np.array([sigma_vl * np.random.randn(), sigma_vw * np.random.randn(), sigma_vh * np.random.randn()])
    s = s.reshape(3, 1)
    shape += s
    return shape.clip(min=min_size, max=max_size)  # [3, 1]


def observe(target, r, sigma_ox, sigma_oy, sigma_oz):
    target = target.reshape(9, 1)
    state, shape = target[:6, :], target[6:, :]
    if state[0, 0] ** 2 + state[2, 0] ** 2 <= r ** 2:
        w = np.array([sigma_ox * np.random.randn(), sigma_oy * np.random.randn(), sigma_oz * np.random.randn()])
        w = w.reshape(3, 1)
        h = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      ])
        z = h @ state
        z += w
        return z, shape  # [3, 1], [3, 1]
    else:
        return None, None