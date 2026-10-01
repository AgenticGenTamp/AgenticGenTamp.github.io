import numpy as np
from env_client import make_env
from geo_approach_old import GeneratedApproach, _handle_offset
from fk import fk
from ik import ik_pose, tcp_for_part_pose, wrap_near

class Prober:
    def __init__(self, seed, pname=None, env=None):
        self.seed=seed; self.env=env or make_env(); self.pname=pname
        self.start()
    def start(self):
        env=self.env
        self.obs,info=env.reset(seed=self.seed)
        ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(self.obs,info)
        self.ap=ap; self.nsteps=0
        if self.pname is None: self.pname=[p for p in ap.parts if 'Tri' in str(self.obs.get_object_from_name(p).type)][0] if any('Tri' in str(self.obs.get_object_from_name(p).type) for p in ap.parts) else ap.parts[0]
        ap.parts=[self.pname]
        ap.goals[self.pname]=np.array([0.3,0.0])
        while True:
            a=ap.get_action(self.obs); self._step(a)
            if ap.stage=='place' and not ap.queue: break
        self.quat=ap._part_pose(self.obs,self.pname)[3:7]; self.gtf=ap.gtf
    def _step(self,a):
        q0=self.ap._robot(self.obs)[1]
        self.obs,r,term,tr,_=self.env.step(a); self.nsteps+=1
        self.term=term; self.tr=tr
        q1=self.ap._robot(self.obs)[1]
        return not(np.allclose(q0,q1) and np.any(a[3:10]!=0))
    def pose(self): return self.ap._part_pose(self.obs,self.pname)
    def goto(self,xyz,grip=0.0,maxstep=0.19):
        """move part to xyz; returns True if all steps accepted"""
        base,q,ga,_=self.ap._robot(self.obs)
        Mt=tcp_for_part_pose(xyz,self.quat,self.gtf)
        qs,pe,_=ik_pose(Mt,q,base=base)
        qs=wrap_near(qs,q); d=qs-q; n=max(1,int(np.ceil(np.max(np.abs(d))/maxstep)))
        ok=True
        for i in range(n):
            a=np.zeros(11,dtype=np.float32); a[3:10]=d/n; a[10]=grip
            if not self._step(a): ok=False; break
        return ok
    def probe(self,x,y,zs=None,zhi=0.17):
        """returns lowest z of part center reached at xy"""
        if self.nsteps>400: self.start()
        p=self.pose()
        if p[2]<zhi-0.001: self.goto([p[0],p[1],zhi])
        self.goto([x,y,zhi])
        p=self.pose(); assert abs(p[0]-x)<1e-3 and abs(p[1]-y)<1e-3 and abs(p[2]-zhi)<1e-3, ('not at hi',p[:3],x,y)
        if zs is None: zs=list(np.arange(0.14,0.0955,-0.003))+[0.095]
        last=self.pose()[2]
        for z in zs:
            if not self.goto([x,y,z]): break
            last=self.pose()[2]
        p=self.pose()
        for z in np.arange(p[2]+0.004,zhi,0.004): self.goto([p[0],p[1],z])
        self.goto([p[0],p[1],zhi])
        assert self.pose()[2]>zhi-0.001, ('lift failed',x,y,last,self.pose()[:3])
        return last

ZS=list(np.arange(0.134,0.0995,-0.002))
def floor_ok(P,x,y):
    return P.probe(x,y,zs=ZS,zhi=0.145)<0.11
def bisect(P,fn_xy,ok_t,bad_t,tol=0.0005):
    """fn_xy(t)->(x,y); ok_t reaches floor, bad_t does not. returns boundary (last ok)"""
    while abs(bad_t-ok_t)>tol:
        m=0.5*(ok_t+bad_t)
        if floor_ok(P,*fn_xy(m)): ok_t=m
        else: bad_t=m
    return ok_t,bad_t
