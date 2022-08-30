import matplotlib.pyplot as plt

import kalman_filter_utils
import tracker_utils
import simulation_utils
import plot_utils


ITER_NUM = 500  # the number of iterations
TIME_INTERVAL = 0.1  # the time interval between frames, second
TARGET_NUM = 30  # the number of targets
GATE = 10  # the association threshold, meter
BLIND_UPDATE_LIMIT = 5  # the limitation of blind update
EGO_LOCATION = [0, 0, 0]  # the location of the ego vehicle
DETECT_RANGE = 75  # the range of detection
SIGMA_AX = 1
SIGMA_AY = 1
SIGMA_AZ = 0.01
SIGMA_OX = 0.1
SIGMA_OY = 0.1
SIGMA_OZ = 0.001
SIGMA_VL = 1
SIGMA_VW = 1
SIGMA_VH = 1
PLOT_RANGE = [-100, -100, -100, 100, 100, 100]


if __name__ == '__main__':
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(projection='3d')
    objs, objs_temp = [], []
    number = 0  # tracking ID
    targets = simulation_utils.init_targets(TARGET_NUM, ITER_NUM)

    for i in range(ITER_NUM):
        print('Iteration:', i + 1)
        
        # control targets' state and shape randomly
        for j in range(TARGET_NUM):
            state = targets[9 * j:9 * j + 6, i]
            targets[9 * j:9 * j + 6, i + 1] = simulation_utils.control_target_state(
                state, TIME_INTERVAL, SIGMA_AX, SIGMA_AY, SIGMA_AZ).reshape(6)
            shape = targets[9 * j + 6:9 * j + 9, i]
            targets[9 * j + 6:9 * j + 9, i + 1] = simulation_utils.control_target_shape(
                shape, SIGMA_VL, SIGMA_VW, SIGMA_VH).reshape(3)

        # get the observation list
        objs_observed = []
        for j in range(TARGET_NUM):
            target = targets[9 * j:9 * j + 9, i + 1]
            state, shape = simulation_utils.observe(target, DETECT_RANGE, SIGMA_OX, SIGMA_OY, SIGMA_OZ)
            if state is not None and shape is not None:
                x, y, z = state[0, 0], state[1, 0], state[2, 0]
                l, w, h = shape[0, 0], shape[1, 0], shape[2, 0]
                obj = tracker_utils.Object(x, y, z, l, w, h)
                objs_observed.append(obj)
        objs_observed_copy = objs_observed.copy()
        
        # associate and track
        num = len(objs)
        for j in range(num):
            flag = False
            idx = 0
            ddm = float('inf')
            
            n = len(objs_observed)
            for k in range(n):
                zx = objs_observed[k].x0
                zy = objs_observed[k].y0
                zz = objs_observed[k].z0
                x = objs[j].tracker.xx[0, 0]
                y = objs[j].tracker.xx[2, 0]
                z = objs[j].tracker.xx[4, 0]
                dd = ((x - zx) ** 2 + (y - zy) ** 2 + (z - zz) ** 2) ** 0.5
                if dd < ddm and dd < GATE:
                    idx = k
                    ddm = dd
                    flag = True
            
            if flag:
                zx = objs_observed[idx].x0
                zy = objs_observed[idx].y0
                zz = objs_observed[idx].z0
                objs[j].tracker.predict()
                objs[j].tracker.update(zx, zy, zz)
                objs[j].tracker_blind_update = 0
                objs_observed.pop(idx)
            else:
                objs[j].tracker.predict()
                objs[j].tracker_blind_update += 1
                
            objs[j].x0 = objs[j].tracker.xx[0, 0]
            objs[j].vx = objs[j].tracker.xx[1, 0]
            objs[j].y0 = objs[j].tracker.xx[2, 0]
            objs[j].vy = objs[j].tracker.xx[3, 0]
            objs[j].z0 = objs[j].tracker.xx[4, 0]
            objs[j].vz = objs[j].tracker.xx[5, 0]
        
        # delete targets which are not updated for a long time
        objs_remained = []
        num = len(objs)
        for j in range(num):
            if objs[j].tracker_blind_update <= BLIND_UPDATE_LIMIT:
                objs_remained.append(objs[j])
        objs = objs_remained
        
        # augment the tracking list
        num = len(objs_temp)
        for j in range(num):
            flag = False
            idx = 0
            ddm = float('inf')
            
            n = len(objs_observed)
            for k in range(n):
                zx = objs_observed[k].x0
                zy = objs_observed[k].y0
                zz = objs_observed[k].z0
                x = objs_temp[j].tracker.xx[0, 0]
                y = objs_temp[j].tracker.xx[2, 0]
                z = objs_temp[j].tracker.xx[4, 0]
                dd = ((x - zx) ** 2 + (y - zy) ** 2 + (z - zz) ** 2) ** 0.5
                if dd < ddm and dd < GATE:
                    idx = k
                    ddm = dd
                    flag = True
            
            if flag:
                zx = objs_observed[idx].x0
                zy = objs_observed[idx].y0
                zz = objs_observed[idx].z0
                x = objs_temp[j].tracker.xx[0, 0]
                y = objs_temp[j].tracker.xx[2, 0]
                z = objs_temp[j].tracker.xx[4, 0]
                
                objs_temp[j].tracker.xx[0, 0] = zx
                objs_temp[j].tracker.xx[1, 0] = (zx - x) / objs_temp[j].tracker.dt
                objs_temp[j].tracker.xx[2, 0] = zy
                objs_temp[j].tracker.xx[3, 0] = (zy - y) / objs_temp[j].tracker.dt
                objs_temp[j].tracker.xx[4, 0] = zz
                objs_temp[j].tracker.xx[5, 0] = (zz - z) / objs_temp[j].tracker.dt
                
                objs_temp[j].x0 = objs_temp[j].tracker.xx[0, 0]
                objs_temp[j].vx = objs_temp[j].tracker.xx[1, 0]
                objs_temp[j].y0 = objs_temp[j].tracker.xx[2, 0]
                objs_temp[j].vy = objs_temp[j].tracker.xx[3, 0]
                objs_temp[j].z0 = objs_temp[j].tracker.xx[4, 0]
                objs_temp[j].vz = objs_temp[j].tracker.xx[5, 0]
                
                objs_observed.pop(idx)
                number += 1
                objs_temp[j].number = number
                objs.append(objs_temp[j])
        
        # augment the temporary tracking list
        objs_temp = objs_observed
        num = len(objs_temp)
        for j in range(num):
            obj = objs_temp[j]
            x0, vx, y0, vy, z0, vz = obj.get_state()
            objs_temp[j].tracker = kalman_filter_utils.KalmanFilter6D(
                TIME_INTERVAL, x0, vx, y0, vy, z0, vz, SIGMA_AX, SIGMA_AY, SIGMA_AZ, SIGMA_OX, SIGMA_OY, SIGMA_OZ)
        
        num = len(objs)
        for j in range(num):
            print('ID:\t', objs[j].number)
            print('xx:\n', objs[j].tracker.xx)
            print('pp:\n', objs[j].tracker.pp)
            print()
        
        # draw dynamically
        ax.clear()
        
        # draw the ego vehicle
        ax.scatter([EGO_LOCATION[0]], [EGO_LOCATION[1]], [EGO_LOCATION[2]], c='blue', s=20)
        
        # draw the range of detection
        xs, ys, zs = plot_utils.get_circle(EGO_LOCATION[0], EGO_LOCATION[1], EGO_LOCATION[2], DETECT_RANGE)
        ax.plot(xs, ys, zs, '--', c='gray', linewidth=1)

        # draw the observation
        num = len(objs_observed_copy)
        for j in range(num):
            obj = objs_observed_copy[j]
            x0, y0, z0, length, width, height = obj.get_box()
            xs, ys, zs, filled = plot_utils.get_voxel(x0, y0, z0, length, width, height)
            ax.voxels(xs, ys, zs, filled, edgecolors='gray', linewidth=0.1, facecolors='gray', alpha=0.5)

        # draw estimated targets and the association gate
        num = len(objs)
        for j in range(num):
            obj = objs[j]
            x0, y0, z0, length, width, height = obj.get_box()
            xs, ys, zs, filled = plot_utils.get_voxel(x0, y0, z0, length, width, height)
            ax.voxels(xs, ys, zs, filled, edgecolors='red', linewidth=0.1, facecolors='red', alpha=0.5)

            xs, ys, zs = plot_utils.get_circle(x0, y0, z0, GATE)
            ax.plot(xs, ys, zs, '--', c='red', linewidth=1)

            text_id = str(objs[j].number)
            ax.text(x0 + length / 2, y0 + width / 2, z0 + height / 2, text_id, color='black', fontsize=6)
        
        # close the drawing window after showing for a while
        xmin, ymin, zmin, xmax, ymax, zmax = PLOT_RANGE
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.tick_params(labelsize=8)
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_zlabel('Z (m)', fontsize=8)
        ax.view_init(elev=45, azim=-135)
        plt.pause(0.02)
        if len(plt.get_fignums()) == 0:
            break
            
    print("\nSimulation process finished!")
