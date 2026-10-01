import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space=action_space
    def reset(self,state,info):
        self.t=0;self.phase=0;self.age=0;self.which=0
        self.objects=[0,54,70];self.retries=0;self.correction=np.zeros(2)
        # Empirically calibrated, constant downward grasp and transfer poses.
        self.qs=np.array([
            [.065525625955,1.3962982300,3.0254953731,-1.3012060674,.26055734398,-.45873585820,1.5707963268],
            [-.012190258209,1.6998111740,3.1705930747,-1.0058878538,-.068036095425,-.43672631857,1.5707963268],
            [0,1.23762613,3.1415926536,-1.3979203,0,-.50604623,1.5707963268],
            [.0015823014293,1.5489995363,3.1420411376,-1.1685636862,-.0010897853438,-.42402965479,1.5707963268]
        ])
        self.beam=np.array(state[38:41])
        self.start_object(state)
    def start_object(self,state):
        self.obj=self.objects[self.which]
        self.pick_position=np.array(state[self.obj:self.obj+2])
        self.b=np.r_[self.pick_position-np.array([.66,0])+self.correction,0.]
        self.target=self.beam.copy();self.target[1]+=[0,-.055,.055][self.which]
        self.phase=0;self.age=0
    def get_action(self,state):
        self.t+=1;self.age+=1
        q=self.qs[[0,1,1,2,2,3,3,2][self.phase]].copy()
        q[6]+=(self.retries//2%2)*(np.pi/2)
        if self.phase in [4,5]:
            self.b[:2]=state[16:18]+np.clip(self.target[:2]-state[self.obj:self.obj+2],-.05,.05)
        ready=np.max(np.abs(q-state[19:26]))<.02 and np.max(np.abs(self.b-state[16:19]))<.004
        advance=(self.age >= (24 if self.phase==2 else 16)) if self.phase in [2,6] else ready or self.age>110
        a=np.zeros(11)
        a[:3]=np.clip(self.b-state[16:19],-.04,.04)
        a[3:10]=np.clip(q-state[19:26],-.1,.1)
        a[10]=float(2<=self.phase<6)
        if advance:
            if self.phase==3 and state[self.obj+2]<.08:
                delta=np.array(state[self.obj:self.obj+2]-self.pick_position)
                delta[1]*=-1
                if np.linalg.norm(delta)<.002:
                    delta=np.array([.008*(-1 if self.retries%2 else 1),-.005])
                self.correction=np.clip(self.correction+.7*np.clip(delta,-.018,.018),-.025,.025)
                self.retries+=1;self.start_object(state)
                return a.astype(np.float32)
            if self.phase<7:self.phase+=1;self.age=0
            elif self.which<2:
                self.which+=1;self.retries=0;self.correction=np.array([0.,.008]);self.start_object(state)
        return a.astype(np.float32)
