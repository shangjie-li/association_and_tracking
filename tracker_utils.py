class Object():
    def __init__(self, x0=None, y0=None, z0=None, length=None, width=None, height=None):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.length = length
        self.width = width
        self.height = height

        self.vx = None
        self.vy = None
        self.vz = None
        
        self.number = None
        self.color = None
        
        self.tracker = None
        self.tracker_blind_update = None

    def get_location(self):
        return self.x0, self.y0, self.z0

    def get_velocity(self):
        return self.vx, self.vy, self.vz

    def get_state(self):
        return self.x0, self.vx, self.y0, self.vy, self.z0, self.vz

    def get_shape(self):
        return self.length, self.width, self.height

    def get_box(self):
        return self.x0, self.y0, self.z0, self.length, self.width, self.height

    def get_state_and_shape(self):
        return self.x0, self.vx, self.y0, self.vy, self.z0, self.vz, self.length, self.width, self.height
