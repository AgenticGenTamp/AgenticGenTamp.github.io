import numpy as np
from scipy.optimize import least_squares
from kinova_candidate import fk

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.mount = .35
        self.tool = .12
    def reset(self, state, info):
        self.t = 0
        self.phase = 0
        self.robot = state.get_objects(self.observation_space.get_type('mujoco_tidybot_robot'))[0]
        self.cubes = [o for o in state.get_objects(self.observation_space.get_type('mujoco_movable_object')) if o.name.startswith('cube_')]
        self.cube = self.cubes[0]
        self.yaw = 2*np.arctan2(state.get(self.cube,"qz"),state.get(self.cube,"qw"))
        self.q = self.ik(.55,.10)
    def ik(self,x,z):
        def fun(v):
            q=np.array([0,v[0],np.pi,v[1],0,v[2],np.pi/2])
            p,r=fk(q,self.tool,(0,0,self.mount))
            return [p[0]-x,p[2]-z,r[0,2],r[2,2]+1]
        v=least_squares(fun,[.1,-1.8,-1.4],max_nfev=100).x
        return np.array([0,v[0],np.pi,v[1],0,v[2],np.pi/2-getattr(self,"yaw",0)])
    def get_action(self, state):
        self.t += 1
        a=np.zeros(18)
        r=self.robot
        base=np.array([state.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
        p=np.array([state.get(self.cube,f) for f in ['x','y','z']])
        if self.t<55:
            target=np.array([p[0]-.65,p[1]-.00135,0])
            a[:3]=np.clip(target-base,-.1,.1)
        if self.t==55:self.q=self.ik(.55,.025)
        if self.t>=85:a[10]=1
        if self.t==100:self.q=self.ik(.55,.55)
        q=np.array([state.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)])
        err=self.q-q
        a[3:10]=np.clip(err,-.1,.1)
        a[11:18]=5*err
        return a.astype(np.float32)
