import numpy as np
from packing import plan_parts
from scipy.spatial.transform import Rotation

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.grasp_q = np.array([0., .391676098, -np.pi, -2.05185294, 0., -.452491313, np.pi/2])
    def reset(self, state, info):
        self.robot = next(o for t in self.observation_space.types for o in state.get_objects(t) if t.name == 'Kinematic3DRobot')
        self.parts = sorted(n for n in state.get_object_names() if n.startswith('part'))
        self.rack = state.get_object_from_name('rack')
        self.done = set()
        self.target = None
        self.last_config=None
        self.motion_stall=0
        self.phase = 'prepick'
        self.phase_steps = 0
        self.slot = 0
        self.vertical=len(self.parts)>2
        if self.vertical:
            self.grasp_q=np.array([0.,.6191014819514105,-np.pi,-1.6,0.,-.9224911716383826,np.pi/2])
        else:
            self.grasp_q=np.array([0.,.391676098,-np.pi,-2.05185294,0.,-.452491313,np.pi/2])
        self.layout = plan_parts(state,[state.get_object_from_name(n) for n in self.parts],self.rack) if len(self.parts)>2 else {}
        if self.layout:
            self.parts=list(self.layout)+[n for n in self.parts if n not in self.layout]
    def xyz(self, state, obj):
        return np.array([state.get(obj, 'pose_'+c) for c in 'xyz'])
    def config(self, state):
        return np.array([state.get(self.robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['joint_'+str(j) for j in range(1,8)]])
    def get_action(self, state):
        a = np.zeros(11, dtype=np.float32)
        q = self.config(state)
        self.motion_stall=self.motion_stall+1 if self.last_config is not None and np.max(np.abs(q-self.last_config))<1e-6 else 0
        self.last_config=q.copy()
        held = state.get(self.robot, 'grasp_active') > .5
        if self.target is None:
            remaining = [n for n in self.parts if n not in self.done]
            if not remaining: return a
            self.target = state.get_object_from_name(remaining[0])
            self.phase = 'prepick'
            self.phase_steps = 0
        pos = self.xyz(state, self.target)
        self.phase_steps += 1
        if self.phase == 'retreat':
            target=self.retreat_q
            if np.max(np.abs(self.difference(target,q)))<.001:
                self.phase='prepick'
            else:
                a[:10]=np.clip(self.difference(target,q),-.2,.2)
                return a
        if self.phase == 'prepick':
            target=self.pick_config(state,pos)
            target[4]-=.3;target[8]-=.3
            if np.max(np.abs(self.difference(target,q)))<.001:
                self.phase='pick'
            else:
                a[:10]=np.clip(self.difference(target,q),-.2,.2)
                return a
        if self.phase == 'pick':
            if held:
                self.phase='lift'; self.phase_steps=0
                self.drop_retry=0
                self.retry_offset=np.zeros(2)
                self.pick_pos=pos.copy()
                rack=self.xyz(state,self.rack)
                idx=self.slot
                self.yaw=0.
                self.drop=rack+np.array([0., -.06 if idx%2==0 else .06, .003])
                if self.target.type.name == 'Kinematic3DTriangle' and state.get(self.target,'triangle_type')>.5:
                    self.drop[:2]-=.05
                if self.target.name in self.layout:
                    origin,self.yaw,_=self.layout[self.target.name]
                    self.drop[:2]=rack[:2]+origin
            else:
                target=self.pick_config(state,pos)
                a[:10]=np.clip(self.difference(target,q),-.2,.2)
                if np.max(np.abs(self.difference(target,q)))<.001: a[10]=-1
                return a
        if self.phase == 'lift':
            target=self.pick_pos.copy(); target[2]=max(self.drop[2]+.09,self.pick_pos[2]+.1)
            if np.linalg.norm(target-pos)<.002:
                self.phase='transfer';self.phase_steps=0
            else:return self.servo(q,pos,target)
        if self.phase == 'transfer':
            target=self.drop.copy();target[2]=max(self.drop[2]+.09,self.pick_pos[2]+.1)
            if np.linalg.norm(target-pos)<.002:
                self.phase='rotate';self.phase_steps=0
            else:return self.servo(q,pos,target)
        if self.phase == 'rotate':
            angle_error=(np.pi/2+self.yaw-q[9]) if self.vertical else (-self.yaw-q[3])
            delta=np.clip((angle_error+np.pi)%(2*np.pi)-np.pi,-.2,.2)
            if abs(delta)<.0001:
                self.phase='lower';self.phase_steps=0
            else:
                if self.vertical:
                    partrot=Rotation.from_quat([state.get(self.target,'pose_q'+c) for c in 'xyzw']).as_matrix()
                    tfrot=Rotation.from_quat([state.get(self.robot,'grasp_tf_q'+c) for c in 'xyzw']).as_matrix()
                    tfpos=np.array([state.get(self.robot,'grasp_tf_'+c) for c in 'xyz'])
                    pivot=(pos-partrot@tfrot.T@tfpos)[:2]
                    angle=delta
                else:
                    pivot=q[:2]+np.array([.1199,0.])
                    angle=-delta
                c,s=np.cos(angle),np.sin(angle)
                newxy=pivot+np.array([[c,-s],[s,c]])@(pos[:2]-pivot)
                a[:2]=np.clip(self.drop[:2]-newxy,-.2,.2)
                a[9 if self.vertical else 3]=delta
                return a
        if self.phase == 'lower':
            if self.vertical and self.motion_stall>=2 and self.drop_retry<8:
                offsets=[(.005,0),(-.005,0),(0,.005),(0,-.005),(.008,0),(-.008,0),(.005,.005),(.005,-.005)]
                offset=np.array(offsets[self.drop_retry])
                self.drop[:2]+=offset-self.retry_offset
                self.retry_offset=offset;self.drop_retry+=1;self.motion_stall=0
                a[:2]=np.clip(self.drop[:2]-pos[:2],-.02,.02)
                return a
            if np.linalg.norm(self.drop-pos)<.001:
                self.phase='release';self.phase_steps=0
                a[10]=1;return a
            return self.servo(q,pos,self.drop)
        if self.phase == 'release':
            if held: a[10]=1;return a
            self.done.add(self.target.name);self.slot+=1
            remaining = [n for n in self.parts if n not in self.done]
            if remaining:
                self.target=state.get_object_from_name(remaining[0])
                self.phase='retreat'
                self.retreat_q=q.copy();self.retreat_q[4]-=.3;self.retreat_q[8]-=.3
            else:self.target=None
            return a
        return a
    def difference(self,target,current):
        delta=target-current
        for i in (2,3,5,7,9):
            delta[i]=(delta[i]+np.pi)%(2*np.pi)-np.pi
        return delta
    def pick_config(self,state,pos):
        kind=self.target.type.name=='Kinematic3DTriangle' and state.get(self.target,'triangle_type')>.5
        if self.vertical:
            grip=np.array([.025,.025]) if kind else np.array([-.005,-.005])
            if self.target.name in self.layout:grip=self.layout[self.target.name][2]
            return np.r_[pos[0]-.61464753+grip[0],pos[1]-.001346857676+grip[1],0,self.grasp_q]
        offset=.02 if kind else 0.
        return np.r_[pos[0]-.53265405+offset,pos[1]-.001+offset,0,self.grasp_q]
    def fk(self, q):
        # Gen3 chain calibrated against held-object poses from the live system.
        translations = [(0,0,.15643),(0,.005375,-.12838),(0,-.21038,-.006375),(0,.006375,-.21038),(0,-.20843,-.006375),(0,0,-.10593),(0,-.10593,0)]
        rx = [np.pi,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2]
        rot=np.eye(3);pos=np.zeros(3)
        for angle,xyz,x in zip(q[3:],translations,rx):
            pos+=rot@np.array(xyz)
            c,s=np.cos(x),np.sin(x)
            ca,sa=np.cos(angle),np.sin(angle)
            rot=rot@np.array([[ca,-sa,0],[c*sa,c*ca,-s],[s*sa,s*ca,c]])
        return pos
    def servo(self,q,pos,target):
        error=target-pos
        start=self.fk(q)
        delta=0.
        for _ in range(4):
            test=q.copy();test[4]+=delta;test[8]+=delta
            end=self.fk(test)
            test[4]+=.001;test[8]+=.001
            dz=(self.fk(test)[2]-end[2])/.001
            if abs(dz)<.02:break
            limit=.06 if (pos[2]<.135 and error[2]>0) or (self.vertical and error[2]<0) else .2
            delta=np.clip(delta+(error[2]-(end[2]-start[2]))/dz,-limit,limit)
        test=q.copy();test[4]+=delta;test[8]+=delta
        shift=self.fk(test)-start
        a=np.zeros(11,dtype=np.float32)
        a[4]=delta;a[8]=delta
        a[0]=np.clip(error[0]-shift[0],-.2,.2)
        a[1]=np.clip(error[1]-shift[1],-.2,.2)
        return a
