import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.a = action_space
        self.o = observation_space
        self.poses = [
            [0, 0, np.pi, .3, 0, 0, 1.57],
            [0, 0, np.pi, .5, 0, -.3, 1.57],
            [0, 0, np.pi, .5, 0, -.6, 1.57],
            [0, -.3, np.pi, .5, 0, 0, 1.57],
            [0, .3, np.pi, .5, 0, 0, 1.57],
        ]

    def reset(self, state, info):
        self.k = 0
        typ = self.o.get_type("mujoco_movable_object")
        rod = sorted(state.get_objects(typ), key=lambda x: x.name)[0]
        self.pick = [float(state.get(rod, "x")), float(state.get(rod, "y"))]

    def get_action(self, state):
        self.k += 1
        a = np.zeros(self.a.shape, dtype=self.a.dtype); a[-1] = 1
        r = state.get_object_from_name("robot")
        if self.k < 16:
            bx=float(state.get(r,"pos_base_x")); by=float(state.get(r,"pos_base_y"))
            a[0]=np.clip((self.pick[0]-.8-bx)/.87,-.1,.1)
            a[1]=np.clip((self.pick[1]-by)/.87,-.1,.1)
        else:
            target=np.array(self.poses[min((self.k-16)//100,4)])
            q=np.array([float(state.get(r,f"pos_arm_joint{i}")) for i in range(1,8)])
            e=(target-q+np.pi)%(2*np.pi)-np.pi
            a[3:10]=np.clip(.4*e,-.1,.1)
        return a
