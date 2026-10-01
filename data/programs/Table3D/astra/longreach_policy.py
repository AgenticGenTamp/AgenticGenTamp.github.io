import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.space = observation_space
        self.rt = observation_space.get_type('Kinematic3DRobot')
        self.ct = observation_space.get_type('Kinematic3DCuboid')
    def reset(self, state, info):
        self.t = 0
        self.r = state.get_objects(self.rt)[0]
        cubes = [o for o in state.get_objects(self.ct) if o.name != 'table']
        self.c = min(cubes, key=lambda o: state.get(o,'pose_x')) if cubes else None
    def get_action(self, state):
        self.t += 1
        a = np.zeros(11, dtype=np.float32)
        if self.c is None: return a
        a[10] = -1
        if state.get(self.r,'grasp_active'):
            a[4] = -0.4
            return a
        if self.t <= 8:
            targets = [state.get(self.c,'pose_x')-.62, state.get(self.c,'pose_y')-.00135, 0, 0, .7, -np.pi, -1.7, 0, -.74159265, np.pi/2]
            fs=['pos_base_x','pos_base_y','pos_base_rot']+['joint_'+str(i) for i in range(1,8)]
            a[:10] = np.clip(np.array(targets)-np.array([state.get(self.r,f) for f in fs]),-.4,.4)
            return a
        # Sweep a set of downward-facing arm configurations across the cube.
        n = (self.t-8) // 16
        q2 = -.9 + .2 * (n % 10)
        q4 = -2.7 + .3 * ((n // 10) % 5)
        q6 = q2-q4-np.pi
        targets = [state.get(self.c,'pose_x')-.65 + .05*(self.t % 16), state.get(self.c,'pose_y'), 0, 0, q2, -np.pi, q4, 0, q6, np.pi/2]
        fs=['pos_base_x','pos_base_y','pos_base_rot']+['joint_'+str(i) for i in range(1,8)]
        a[:10] = np.clip(np.array(targets)-np.array([state.get(self.r,f) for f in fs]),-.4,.4)
        return a
