import numpy as np
from scipy.optimize import least_squares
from kinova_candidate import fk

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space=action_space
        self.observation_space=observation_space
        self.mount=.35
        self.tool=.12
        self.cache={}
    def ik(self,x,z):
        key=(x,z)
        if key not in self.cache:
            def fun(v):
                q=np.array([0,v[0],np.pi,v[1],0,v[2],np.pi/2])
                p,r=fk(q,self.tool,(0,0,self.mount))
                return [p[0]-x,p[2]-z,r[0,2],r[2,2]+1]
            v=least_squares(fun,[.1,-1.8,-1.4],max_nfev=100).x
            self.cache[key]=np.array([0,v[0],np.pi,v[1],0,v[2],np.pi/2])
        q=self.cache[key].copy();q[6]-=getattr(self,'yaw',0)
        return q
    def xyz(self,state,obj):
        return np.array([state.get(obj,f) for f in ['x','y','z']])
    def reset(self,state,info):
        self.robot=state.get_objects(self.observation_space.get_type('mujoco_tidybot_robot'))[0]
        movable=state.get_objects(self.observation_space.get_type('mujoco_movable_object'))
        self.cubes=sorted([o for o in movable if o.name.startswith('cube_')],key=lambda o:o.name)
        self.bins=[o for o in movable if o.name.startswith('bin_')]
        self.done=set();self.tries={};self.cube=None;self.phase='select';self.t=0
        self.yaw=0;self.q=self.ik(.55,.55)
    def transition(self,phase):
        self.phase=phase;self.t=0
    def get_action(self,state):
        r=self.robot
        base=np.array([state.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
        q=np.array([state.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)])
        a=np.zeros(18);g=0;target=None
        if self.phase=='select':
            pending=[c for c in self.cubes if c.name not in self.done]
            if not pending:return a.astype(np.float32)
            self.cube=min(pending,key=lambda c:self.tries.get(c.name,0)*10+np.linalg.norm(self.xyz(state,c)[:2]-base[:2]))
            self.bin=min(self.bins,key=lambda b:np.linalg.norm(self.xyz(state,b)[:2]-self.xyz(state,self.cube)[:2]))
            self.tries[self.cube.name]=self.tries.get(self.cube.name,0)+1
            index=self.cubes.index(self.cube)
            self.aim_y=0 if len(self.cubes)==1 else .055*(2*(index%2)-1)
            yaw=2*np.arctan2(state.get(self.cube,'qz'),state.get(self.cube,'qw'))
            while yaw>np.pi/4:yaw-=np.pi/2
            while yaw< -np.pi/4:yaw+=np.pi/2
            self.yaw=yaw if yaw<-.4 or yaw>.4 else 0
            if self.tries[self.cube.name]%3==2:self.yaw=yaw
            self.q=self.ik(.55,.10);self.transition('hover')
        p=self.xyz(state,self.cube);b=self.xyz(state,self.bin)
        if self.phase=='hover':
            target=np.array([p[0]-.65,p[1]-.00135,0])
            if self.t>=30 and np.linalg.norm(target[:2]-base[:2])<.02:
                self.q=self.ik(.55,.025);self.transition('descend')
        elif self.phase=='descend':
            if self.t>=20:self.transition('close')
        elif self.phase=='close':
            g=1
            if self.t>=8:self.q=self.ik(.55,.65);self.transition('lift')
        elif self.phase=='lift':
            g=1
            if self.t>=18:
                if p[2]<.25:self.transition('select')
                else:
                    self.yaw=0;self.q=self.ik(.4,.60);self.transition('carry')
        elif self.phase=='carry':
            g=1
            if self.t>15 and p[2]<.25:
                self.transition('select')
                return np.zeros(18,dtype=np.float32)
            # Aim using the actual held object, compensating grasp offsets.
            target=np.array([b[0]-2.43,b[1]+self.aim_y-.00135,0])
            if self.t>20:
                tool_z=fk(q,self.tool,(.1,0,self.mount))[0][2]
                grip_dz=p[2]-tool_z
                throw_distance=1.93-5.8*(grip_dz+.006)
                target[:2]=base[:2]+np.array([b[0]-throw_distance,b[1]+self.aim_y])-p[:2]
            target[0]=min(target[0],.985)
            if self.t>=40 and np.max(np.abs(target-base))<.003 and np.max(np.abs(self.q-q))<.006:
                self.transition('kick')
        elif self.phase=='kick':
            g=1;a[[12,14,16]]=[0,12,12]
            self.transition('release');self.t+=1;a[10]=g
            return a.astype(np.float32)
        elif self.phase=='release':
            a[[12,14,16]]=[0,12,12]
            self.transition('flight');self.t+=1
            return a.astype(np.float32)
        elif self.phase=='flight':
            if self.t>=18:
                if np.linalg.norm(p[:2]-b[:2])<.24:self.done.add(self.cube.name)
                elif p[0]>1.9:self.done.add(self.cube.name)
                self.transition('select')
        err=self.q-q
        a[3:10]=np.clip(err,-.1,.1);a[11:]=5*err;a[10]=g
        if target is not None:a[:3]=np.clip(target-base,-.1,.1)
        self.t+=1
        return a.astype(np.float32)
