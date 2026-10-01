import numpy as np
import kinova, ctrl

class Runner:
    """Shared control core; also used (mirrored) by approach.py."""
    def __init__(self, env=None, seed=0, obs=None):
        if env is not None:
            self.env=env
            self.obs,_=env.reset(seed=seed)
        else:
            self.env=None; self.obs=obs
        self.steps=0
        self.q_des=self.obs[19:26].copy()
        self.qi=np.zeros(7)
        self.setp=ctrl.ee_world(self.obs)
        self.grip=0.0
        self.rews=[]; self.term_step=None
    def step(self,a):
        self.obs,r,te,tr,_=self.env.step(np.asarray(a,dtype=np.float32))
        self.steps+=1; self.rews.append(r)
        if te and self.term_step is None: self.term_step=self.steps
        return r,te
    def arm_action(self, ff=None):
        err=self.q_des-self.obs[19:26]
        self.qi=np.clip(self.qi+0.04*err,-0.3,0.3)
        if ff is None: ff=np.zeros(7)
        return np.clip(4.0*(ff+0.5*err+self.qi),-0.1,0.1)
    def base_action(self, base, vmax=0.1):
        a=np.zeros(3)
        if base is None: return a
        a[0]=np.clip((base[0]-self.obs[16])/ctrl.BASE_GAIN,-vmax,vmax)
        a[1]=np.clip((base[1]-self.obs[17])/ctrl.BASE_GAIN,-vmax,vmax)
        a[2]=np.clip((base[2]-self.obs[18])/ctrl.YAW_GAIN,-0.1,0.1)
        return a
    def go(self, goal, base=None, rel=False, fyaw=None, speed=0.03, tol=0.004,
           maxsteps=200, settle=4, bvmax=0.1, verbose=False):
        """goal: world xyz, or base-relative (dx,dy,z) if rel=True."""
        goal=np.asarray(goal,float); done=0
        for i in range(maxsteps):
            b=self.obs[16:19]
            g = np.array([b[0]+goal[0], b[1]+goal[1], goal[2]]) if rel else goal
            zd=np.array([0,0,-1.0])
            yd=None if fyaw is None else ctrl.Rz(b[2]).T@np.array([np.cos(fyaw),np.sin(fyaw),0.0])
            d=g-self.setp; n=np.linalg.norm(d)
            self.setp = self.setp+d/n*speed if n>speed else g.copy()
            dq=kinova.ik_step(self.q_des, ctrl.w2a(self.setp,b), zd,
                              tool_offset=ctrl.TOOL, yaw_des=yd)
            sd=np.clip(dq,-0.024,0.024); self.q_des=self.q_des+sd
            a=np.zeros(11); a[3:10]=self.arm_action(sd); a[10]=self.grip
            a[0:3]=self.base_action(base,bvmax)
            self.step(a)
            bok = base is None or (abs(base[0]-self.obs[16])<2e-3 and abs(base[1]-self.obs[17])<2e-3)
            if n<1e-6 and bok and np.linalg.norm(g-ctrl.ee_world(self.obs))<tol:
                done+=1
                if done>=settle: break
        return ctrl.ee_world(self.obs), i
    def set_grip(self,g,n=14):
        self.grip=g
        for i in range(n):
            a=np.zeros(11); a[3:10]=self.arm_action(); a[10]=g; self.step(a)
