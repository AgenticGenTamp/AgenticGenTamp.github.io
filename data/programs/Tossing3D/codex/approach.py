"""Collision-based policy for variable-count Tossing3D."""
import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.pre_q = np.array([.9441,-.0616,1.4739,1.3419,-.5038,-1.5938,.2053])
        self.offset = np.array([.6,.25])
        self.sweep = np.array([
[-.00005465,.00003504,.01686388,.1,.1,.1,.1,-.1,-.1,.1,1,3.693535,4.136233,8.466327,10.530969,-8.755755,-2.834353,9.827700],
[.00000791,.00003101,-.07536265,-.1,.1,-.1,-.1,.1,.1,.1,1,-12,12,-12,-4.223864,12,12,11.555738],
[.00013387,.00000992,-.08682545,-.1,.1,-.1,-.1,.1,.1,.1,1,-12,12,-12,-1.579574,11.547497,12,8.361640],
[-.00000251,.00011905,.06300577,-.1,.1,-.1,-.1,.1,.1,.1,1,-12,12,-12,-1.868824,8.332363,12,5.939990],
[-.00012625,.00007193,-.06538130,-.1,.1,-.1,-.061172,.1,.1,.1,1,-12,12,-12,-.489376,5.980481,12,4.221514],
[-.00013797,-.00004262,-.00062417,-.1,.1,-.1,-.1,.1,.1,.1,1,-12,10.235282,-10.266092,-.817169,4.211204,12,2.906250]],dtype=np.float32)
        self.reset(None,None)

    def reset(self,state,info):
        self.phase="setup"; self.n=0; self.i=0; self.cycles=0; self.cube_name=None
        self.done=set(); self.alpha=0.0; self.around_end=False

    def _v(self,state,name,features):
        o=state.get_object_from_name(name)
        return np.array([float(state.get(o,f)) for f in features])

    def _xyz(self,state,name): return self._v(state,name,("x","y","z"))

    def _robot(self,state):
        b=self._v(state,"robot",("pos_base_x","pos_base_y","pos_base_rot"))
        q=self._v(state,"robot",tuple("pos_arm_joint%d"%i for i in range(1,8)))
        return b,q

    def _placed(self,state,cube,bin_name):
        p,g=self._xyz(state,cube),self._xyz(state,bin_name)
        b=state.get_object_from_name(bin_name)
        hx=max(.06,.5*float(state.get(b,"bb_x"))); hy=max(.06,.5*float(state.get(b,"bb_y")))
        return abs(p[0]-g[0])<hx and abs(p[1]-g[1])<hy and p[2]>.06

    def get_action(self,state):
        names=state.get_object_names(); cubes=[n for n in names if n.startswith("cube_")]
        bins=[n for n in names if n.startswith("bin_")]
        if not cubes or not bins: return np.zeros(self.action_space.shape,dtype=np.float32)
        bn=bins[0]
        for c in cubes:
            if self._placed(state,c,bn): self.done.add(c)
        if self.cube_name is None or self.cube_name in self.done:
            left=[c for c in cubes if c not in self.done]
            if not left: return np.zeros(self.action_space.shape,dtype=np.float32)
            b,_=self._robot(state); self.cube_name=min(left,key=lambda c:np.linalg.norm(self._xyz(state,c)[:2]-b[:2]))
            self.phase="setup"; self.n=0; self.cycles=0
            self.around_end=False
        cube=self._xyz(state,self.cube_name); goal=self._xyz(state,bn)
        # Contact is highly non-linear in yaw.  These calibrated orientations
        # reliably push toward the corresponding end of the barrier.  Mirror
        # the base approach for bins below the barrier rather than always
        # routing every cube toward positive world Y.
        if goal[1] < 0 and cube[1] < -1.45:
            self.around_end=True
        self.alpha=1.305 if goal[1] < 0 and not self.around_end else -1.305
        ca,sa=np.cos(self.alpha),np.sin(self.alpha); rot=np.array([[ca,-sa],[sa,ca]])
        if self.phase=="setup":
            b,q=self._robot(state); bt=cube[:2]-rot.dot(self.offset); yt=.4799+self.alpha
            a=np.zeros(self.action_space.shape,dtype=np.float32); a[:2]=np.clip(bt-b[:2],-.06,.06)
            ye=(yt-b[2]+np.pi)%(2*np.pi)-np.pi; a[2]=np.clip(ye,-.06,.06)
            # Preserve the empirically discovered winding branch; taking the
            # shortest angular wrap sends joint 4 to a different configuration.
            qe=self.pre_q-q; a[3:10]=np.clip(qe,-.06,.06); a[10]=1.; a[11:18]=np.clip(4*qe,-4,4)
            self.n+=1
            limit=55 if self.cycles==0 else 32
            if (np.linalg.norm(bt-b[:2])<.018 and abs(ye)<.025 and np.max(np.abs(qe))<.035) or self.n>=limit:
                self.phase="strike"; self.i=0; self.n=0
            return a
        if self.phase=="strike":
            a=self.sweep[self.i].copy(); a[:2]=rot.dot(a[:2])
            # The 0.8 sweep produces strong lateral travel; switch to the
            # forward-biased calibrated sweep near the wall's end.
            if abs(cube[1])<1.65: a[11:18]*=.8
            a[10]=1.; self.i+=1
            if self.i==len(self.sweep): self.phase="coast"; self.n=0
            return a
        a=np.zeros(self.action_space.shape,dtype=np.float32); a[10]=1.; self.n+=1
        if self.n>=2: self.phase="setup"; self.n=0; self.cycles+=1
        return a
