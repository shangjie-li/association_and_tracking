import kalman_filter_utils


class Object():
    def __init__(self, x0=None, y0=None, z0=None, l0=None, w0=None, h0=None):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.l0 = l0
        self.w0 = w0
        self.h0 = h0

        self.vx = None
        self.vy = None
        self.vz = None
        
        self.number = None
        self.color = None
        
        self.tracker = None
        self.tracker_blind_update = None
        self.tracker_confirmed_times = None

        self.smoother_l0 = None
        self.smoother_w0 = None
        self.smoother_h0 = None

    def get_location(self):
        return self.x0, self.y0, self.z0

    def get_velocity(self):
        return self.vx, self.vy, self.vz

    def get_state(self):
        return self.x0, self.vx, self.y0, self.vy, self.z0, self.vz

    def get_shape(self):
        return self.l0, self.w0, self.h0

    def get_box(self):
        return self.x0, self.y0, self.z0, self.l0, self.w0, self.h0

    def get_state_and_shape(self):
        return self.x0, self.vx, self.y0, self.vy, self.z0, self.vz, self.l0, self.w0, self.h0

    def update_state_from_tracker(self):
        if self.tracker is not None:
            x0, vx, y0, vy, z0, vz = self.tracker.get_state()
            self.x0 = x0
            self.y0 = y0
            self.z0 = z0
            self.vx = vx
            self.vy = vy
            self.vz = vz
        if self.smoother_l0 is not None:
            self.l0 = self.smoother_l0.get_location()
        if self.smoother_w0 is not None:
            self.w0 = self.smoother_w0.get_location()
        if self.smoother_h0 is not None:
            self.h0 = self.smoother_h0.get_location()

    def limit_shape(self, min_size, max_size):
        self.l0 = min_size if self.l0 < min_size else self.l0
        self.l0 = max_size if self.l0 > max_size else self.l0
        self.w0 = min_size if self.w0 < min_size else self.w0
        self.w0 = max_size if self.w0 > max_size else self.w0
        self.h0 = min_size if self.h0 < min_size else self.h0
        self.h0 = max_size if self.h0 > max_size else self.h0

    def make_tracker_predict(self):
        self.tracker.predict()
        if self.smoother_l0 is not None:
            self.smoother_l0.predict()
        if self.smoother_w0 is not None:
            self.smoother_w0.predict()
        if self.smoother_h0 is not None:
            self.smoother_h0.predict()

    def make_tracker_update(self, zx, zy, zz, zl=None, zw=None, zh=None):
        self.tracker.update(zx, zy, zz)
        if zl is not None and self.smoother_l0 is not None:
            self.smoother_l0.update(zl)
        if zw is not None and self.smoother_w0 is not None:
            self.smoother_w0.update(zw)
        if zh is not None and self.smoother_h0 is not None:
            self.smoother_h0.update(zh)


class MultipleTargetTracker():
    def __init__(self, dt, sigma_ax=1, sigma_ay=1, sigma_az=1, sigma_ox=1, sigma_oy=1, sigma_oz=1,
                 gate=10, blind_update_limit=5, confirmation_thresh=5, min_size=1.0, max_size=10.0):
        self.dt = dt
        self.sigma_ax = sigma_ax
        self.sigma_ay = sigma_ay
        self.sigma_az = sigma_az
        self.sigma_ox = sigma_ox
        self.sigma_oy = sigma_oy
        self.sigma_oz = sigma_oz
        self.gate = gate
        self.blind_update_limit = blind_update_limit
        self.confirmation_thresh = confirmation_thresh

        self.min_size = min_size
        self.max_size = max_size

        self.objs_temp = []
        self.objs = []
        self.tracked_num = 0

    def find_nearest_object(self, obj, objs_observed):
        flag, idx, ddm = False, 0, float('inf')
        for k in range(len(objs_observed)):
            zx, zy, zz = objs_observed[k].get_location()
            x, y, z = obj.tracker.get_location()
            dd = ((x - zx) ** 2 + (y - zy) ** 2 + (z - zz) ** 2) ** 0.5
            if dd < ddm and dd < self.gate:
                flag, idx, ddm = True, k, dd
        return flag, idx, ddm

    def update_objects(self, inputs):
        # associate and track
        objs_observed = inputs.copy()
        for j in range(len(self.objs)):
            flag, idx, _ = self.find_nearest_object(self.objs[j], objs_observed)
            if flag:
                zx, zy, zz, zl, zw, zh = objs_observed[idx].get_box()
                self.objs[j].make_tracker_predict()
                self.objs[j].make_tracker_update(zx, zy, zz, zl, zw, zh)
                self.objs[j].tracker_blind_update -= 1 if self.objs[j].tracker_blind_update > 0 else 0
                self.objs[j].tracker_confirmed_times += 1
                objs_observed.pop(idx)
            else:
                self.objs[j].make_tracker_predict()
                self.objs[j].tracker_blind_update += 1
                self.objs[j].tracker_confirmed_times -= 1 if self.objs[j].tracker_confirmed_times > 0 else 0
            self.objs[j].update_state_from_tracker()
            self.objs[j].limit_shape(min_size=self.min_size, max_size=self.max_size)

        # delete targets which are not updated for a long time
        objs_remained = []
        for j in range(len(self.objs)):
            if self.objs[j].tracker_blind_update <= self.blind_update_limit:
                objs_remained.append(self.objs[j])
        self.objs = objs_remained

        # augment the tracking list
        for j in range(len(self.objs_temp)):
            flag, idx, _ = self.find_nearest_object(self.objs_temp[j], objs_observed)
            if flag:
                zx, zy, zz, zl, zw, zh = objs_observed[idx].get_box()
                x, y, z = self.objs_temp[j].tracker.get_location()
                vx, vy, vz = (zx - x) / self.dt, (zy - y) / self.dt, (zz - z) / self.dt
                self.objs_temp[j].tracker.set_state(zx, vx, zy, vy, zz, vz)
                self.tracked_num += 1
                self.objs_temp[j].number = self.tracked_num
                self.objs_temp[j].tracker_blind_update = 0
                self.objs_temp[j].tracker_confirmed_times = 0
                self.objs_temp[j].smoother_l0 = kalman_filter_utils.KalmanFilter2D(self.dt, zl, 0, 1, 1)
                self.objs_temp[j].smoother_w0 = kalman_filter_utils.KalmanFilter2D(self.dt, zw, 0, 1, 1)
                self.objs_temp[j].smoother_h0 = kalman_filter_utils.KalmanFilter2D(self.dt, zh, 0, 1, 1)
                self.objs_temp[j].update_state_from_tracker()
                self.objs.append(self.objs_temp[j])
                objs_observed.pop(idx)

        # augment the temporary tracking list
        self.objs_temp = objs_observed
        for j in range(len(self.objs_temp)):
            x0, vx, y0, vy, z0, vz = self.objs_temp[j].get_state()
            self.objs_temp[j].tracker = kalman_filter_utils.KalmanFilter6D(
                self.dt, x0, vx, y0, vy, z0, vz,
                self.sigma_ax, self.sigma_ay, self.sigma_az, self.sigma_ox, self.sigma_oy, self.sigma_oz)

    def get_confirmed_objects(self):
        objs_confirmed = []
        for j in range(len(self.objs)):
            if self.objs[j].tracker_confirmed_times >= self.confirmation_thresh:
                objs_confirmed.append(self.objs[j])
        return objs_confirmed
