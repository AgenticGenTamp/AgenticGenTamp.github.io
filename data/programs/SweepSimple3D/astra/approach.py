import numpy as np

class GeneratedApproach:
    """Feedback-controlled floor collection using the mobile arm."""
    def __init__(self, action_space, observation_space, primitives):
        self.action_space=action_space
        self.observation_space=observation_space
        self.low=np.asarray(action_space.low)
        self.high=np.asarray(action_space.high)
    def reset(self,state,info):
        self.robot=state.get_objects(self.observation_space.get_type('mujoco_tidybot_robot'))[0]
        self.cubes=list(state.get_objects(self.observation_space.get_type('mujoco_movable_object')))
        self.cubes=[o for o in self.cubes if o.name.startswith('cube_')]
        self.fields=['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint%d'%j for j in range(1,8)]
        self.q=np.array([0,2.2418,np.pi,.3,0,-.873,np.pi/2])
        self.phase='choose';self.t=0;self.round=0;self.done=set();self.target=None
        self.goals=[(.5,1.5,-np.pi),(1.5,1.8,np.pi/2),(2.1,1.5,0),(1.5,-.5,-np.pi/2)]
    def get_action(self,state):
        a=np.zeros(11,dtype=np.float32)
        if not self.cubes:return a
        v=np.array([state.get(self.robot,f) for f in self.fields])
        def xy(o):return np.array([state.get(o,'x'),state.get(o,'y')])
        if self.phase=='choose':
            remaining=[o for o in self.cubes if o.name not in self.done]
            if not remaining:
                self.done.clear();self.round+=1;remaining=self.cubes
            self.target=max(remaining,key=lambda o:state.get(o,'x'))
            self.phase='approach';self.t=0
            self.heading=-np.pi/2 if xy(self.target)[1]<1.1 else np.pi/2
        c=xy(self.target);goal=np.array(self.goals[self.round%len(self.goals)])
        if self.phase in ('stage','turn','deliver') and c[0]<.94 and -.95<c[1]<1.2 and state.get(self.target,'z')<.06:
            self.phase='release';self.t=0
        q=self.q.copy();grip=1
        if self.phase=='approach':
            grip=0
            base=np.r_[c-.85*np.array([np.cos(self.heading),np.sin(self.heading)]),self.heading]
            if (np.max(abs(q-v[3:]))<.045 and np.linalg.norm(base[:2]-v[:2])<.012 and abs((self.heading-v[2]+np.pi)%(2*np.pi)-np.pi)<.02 and self.t>8) or self.t>=155:
                self.phase='close';self.t=0
        elif self.phase=='close':
            base=np.r_[c-.85*np.array([np.cos(self.heading),np.sin(self.heading)]),self.heading]
            if self.t>=8:self.phase='stage';self.t=0
        elif self.phase=='stage':
            base=np.array([1.7,1.45,self.heading])
            if np.linalg.norm(base[:2]-v[:2])<.025 or self.t>100:
                self.phase='turn';self.t=0
        elif self.phase=='turn':
            base=np.array([1.7,1.45,goal[2]])
            if abs((goal[2]-v[2]+np.pi)%(2*np.pi)-np.pi)<.025 or self.t>135:
                self.phase='deliver';self.t=0
        elif self.phase=='deliver':
            base=v[:3].copy();base[:2]+=np.clip(goal[:2]-c,-.025,.025);base[2]=goal[2]
            if np.linalg.norm(c-goal[:2])<.012 or self.t>120:
                self.phase='release';self.t=0
        else:
            base=v[:3].copy();grip=0
            if self.t>6:
                self.done.add(self.target.name);self.phase='choose';self.t=0
        delta=base-v[:3];delta[2]=(delta[2]+np.pi)%(2*np.pi)-np.pi
        speed=.065 if self.phase=='approach' else .025
        a[:3]=np.clip(delta,-speed,speed)
        a[3:10]=np.clip(q-v[3:],-.1,.1);a[10]=grip
        self.t+=1
        return np.clip(a,self.low,self.high).astype(np.float32)
