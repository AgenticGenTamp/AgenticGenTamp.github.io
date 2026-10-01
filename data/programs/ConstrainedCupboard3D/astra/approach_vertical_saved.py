import numpy as np
from scipy.spatial.transform import Rotation
from kin import ik, fk, planar_ik

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space=action_space
        self.space=observation_space
        self.movetype=observation_space.get_type('mujoco_movable_object')
        self.fixtype=observation_space.get_type('mujoco_fixture')
    def reset(self,state,info):
        self.robot=state.get_object_from_name('robot')
        self.objects=sorted(state.get_objects(self.movetype),key=lambda o:o.name)
        self.fixtures=sorted(state.get_objects(self.fixtype),key=lambda o:o.name)
        self.qt=self.joints(state)
        self.index=0;self.stage=0;self.queue=[];self.wait=0;self.lastq=None;self.waysteps=0
        self.plan_pick(state)
    def joints(self,s):
        return np.array([s.get(self.robot,'pos_arm_joint'+str(j)) for j in range(1,8)])
    def xyz(self,s,o):return np.array([s.get(o,f) for f in ('x','y','z')])
    def add(self,p,rot,base,grip,wait=5):
        q0=self.lastq if self.lastq is not None else np.array([0,1.7,np.pi,-1.4,0,-.4,0])
        local=Rotation.from_euler('z',-base[2]).apply(np.asarray(p)-np.array([base[0],base[1],.4]))-np.array([.12,0,0])
        localrot=Rotation.from_euler('z',-base[2]).as_matrix()@rot
        yaw=np.arctan2(localrot[1,0],localrot[0,0])
        q,err=planar_ik(local[0],local[2],yaw=yaw,q0=q0)
        self.lastq=q
        self.queue.append((q,np.asarray(base).copy(),grip,wait))
    def plan_pick(self,s):
        if self.index>=len(self.objects):
            self.stage=3;return
        obj=self.objects[self.index];p=self.xyz(s,obj)
        quat=[s.get(obj,f) for f in ('qx','qy','qz','qw')]
        yaw=Rotation.from_quat(quat).as_euler('xyz')[2]
        yaw=(yaw+np.pi/2)%np.pi-np.pi/2
        p=p+np.array([-np.sin(yaw),np.cos(yaw),0.]) *.10
        rot=Rotation.from_euler('z',yaw).as_matrix()@np.diag([1.,-1.,-1.])
        base=np.array([p[0]-.52,p[1],0.])
        self.lastq=None
        self.add([p[0],p[1],.2],rot,base,0)
        self.add([p[0],p[1],.005],rot,base,0)
        self.add([p[0],p[1],.005],rot,base,1,8)
        self.add([p[0],p[1],.3],rot,base,1)
        self.stage=1
    def plan_place(self,s):
        fixture=self.fixtures[(1+2*self.index)%len(self.fixtures)]
        obj=self.objects[self.index]
        qactual=self.joints(s);fp,fr=fk(qactual)
        robotbase=np.array([s.get(self.robot,'pos_base_x'),s.get(self.robot,'pos_base_y'),.4])
        held=fr.T@(self.xyz(s,obj)-robotbase-np.array([.12,0,0])-fp)
        self.held=held
        target=self.xyz(s,fixture)+np.array([0,0,.2])
        pitch=np.pi/2;yaw=np.pi/2
        _,rotation=fk(planar_ik(.65,-.1,yaw,pitch)[0])
        tool=target-rotation@held
        for pos in [tool+[-.5,0,0],tool]:
            q,err=planar_ik(.65,pos[2]-.4,yaw,pitch)
            base=np.array([pos[0]-.77,pos[1]-.00135,0.])
            self.queue.append((q,base,1,5))
        self.queue.append((q,base.copy(),0,8))
        self.queue.append((q,base+[-.4,0,0],0,5))
        self.stage=2
    def get_action(self,state):
        if not self.queue:
            if self.stage==1:self.plan_place(state)
            elif self.stage==2:
                self.index+=1;self.plan_pick(state)
        a=np.zeros(11,dtype=np.float32)
        if not self.queue:return a
        q,base,grip,settle=self.queue[0]
        self.waysteps+=1
        current=np.array([state.get(self.robot,f) for f in ('pos_base_x','pos_base_y','pos_base_rot')])
        a[:3]=np.clip((base-current)/[.87,.87,1],-.1,.1)
        a[3:10]=np.clip(2*(q-self.joints(state)),-.1,.1)
        self.qt+=.25*a[3:10];a[10]=grip
        if np.max(np.abs(q-self.joints(state)))<.04 and np.linalg.norm(base-current)<.005:
            self.wait+=1
            if self.wait>=settle:self.queue.pop(0);self.wait=0;self.waysteps=0
        else:self.wait=0
        if self.waysteps>(60 if self.stage==2 else 180):
            self.queue.pop(0);self.wait=0;self.waysteps=0
        return a
