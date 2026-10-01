"""Probe library built on kinlib (calibrated)."""
import numpy as np
import kinlib
from kinlib import RDOWN, linpos, ik, rotz

JN=['joint_%d'%i for i in range(1,8)]
GRASP_D=np.array([-0.015,0.0,0.0])

def rob(o): return o.get_object_from_name('robot')
def q_of(o):
    r=rob(o); return np.array([o.get(r,n) for n in JN])
def base(o):
    r=rob(o); return np.array([o.get(r,'pos_base_x'),o.get(r,'pos_base_y'),0.0])
def grasping(o): return o.get(rob(o),'grasp_active')>0.5
def feat(o,name):
    x=o.get_object_from_name(name); return {k:o.get(x,k) for k in o.type_features[x.type]}
def ppos(o,name):
    x=o.get_object_from_name(name)
    return np.array([o.get(x,f) for f in ('pose_x','pose_y','pose_z')])
def yaw(o,name):
    x=o.get_object_from_name(name)
    return 2*np.arctan2(o.get(x,'pose_qz'),o.get(x,'pose_qw'))
def lp(o):
    o=getattr(o,'obs',o)
    return linpos(q_of(o))+base(o)
def parts(o): return sorted([n for n in o.get_object_names() if n.startswith('part')],key=lambda s:int(s[4:]))

class Drv:
    def __init__(self,env):
        self.env=env; self.rng=np.random.default_rng(0)
        self.obs=None; self.term=False; self.trunc=False; self.steps=0
    def reset(self,seed=0,oc=None):
        opt={'object_count':oc} if oc else None
        self.obs,info=self.env.reset(seed=seed,options=opt)
        self.term=False;self.trunc=False;self.steps=0
        return self.obs
    def step(self,a):
        self.obs,r,t,tr,info=self.env.step(np.asarray(a,dtype=np.float32))
        self.term=self.term or t; self.trunc=tr; self.steps+=1
        return self.obs
    def act(self,dq,grip=0.0):
        a=np.zeros(11); a[3:10]=np.clip(dq,-0.2,0.2); a[10]=grip; return a
    def grip(self,v):
        a=np.zeros(11); a[10]=v; self.step(a)
    def move(self,target_fn,R=RDOWN,step=0.03,tol=3e-3,grip=0.0,maxs=200,stop_fn=None):
        blocked=0
        for _ in range(maxs):
            o=self.obs
            if stop_fn is not None and stop_fn(o): return 'stop'
            q=q_of(o); b=base(o); p=linpos(q)+b
            tgt=np.asarray(target_fn(o),float)
            d=tgt-p; n=np.linalg.norm(d)
            if n<tol: return 'ok'
            w=p+d*min(1.0,step/n)
            qd,ok=ik(q,w-b,R,restarts=4,rng=self.rng)
            if not ok:
                qd,ok=ik(q,tgt-b,R,restarts=6,rng=self.rng)
                if not ok: return 'ikfail'
            dq=qd-q
            if np.max(np.abs(dq))<1e-6: return 'ok'
            self.step(self.act(dq,grip))
            if np.max(np.abs(q_of(self.obs)-q))<1e-9:
                blocked+=1
                if blocked>=2: return 'blocked'
            else: blocked=0
        return 'slow'

def ref_c(o,name):
    f=feat(o,name)
    if 'triangle_type' in f and f['triangle_type']>0.5:
        return np.array([f['side_a']/3.,f['side_b']/3.,0.])
    return np.zeros(3)

def grasp(d,name):
    o=d.obs; f=feat(o,name); p=ppos(o,name); c=ref_c(o,name)
    tgt=p+rotz(yaw(o,name))@c+RDOWN@GRASP_D
    r=d.move(lambda s,h=tgt+np.array([0,0,0.15]):h,step=0.05,tol=0.01)
    if r=='ikfail': return False,'pre_ikfail'
    r=d.move(lambda s,t=tgt:t,step=0.02,tol=2e-3)
    d.grip(-1.0)
    if grasping(d.obs): return True,'ok'
    for dz in (0.01,-0.01,0.02):
        d.move(lambda s,t=tgt+np.array([0,0,dz]):t,step=0.01,tol=2e-3)
        d.grip(-1.0)
        if grasping(d.obs): return True,'ok%+.2f'%dz
    return False,'grasp_fail:'+r

def part_ref(d,name):
    return ppos(d.obs,name)+rotz(yaw(d.obs,name))@ref_c(d.obs,name)

def lin_for(d,name,T):
    off=part_ref(d,name)-lp(d)
    return np.asarray(T,float)-off

def place(d,name,x,y,zhi=0.20,zlo=0.085,dz=0.0025,carry=0.32):
    """grasp already done. carry to (x,y) then descend ref point, releasing."""
    pr=part_ref(d,name)
    r=d.move(lambda s:lin_for(d,name,[pr[0],pr[1],carry]),step=0.04,tol=5e-3)
    if r!='ok': return 'lift:'+r
    r=d.move(lambda s:lin_for(d,name,[x,y,carry]),step=0.04,tol=5e-3)
    if r!='ok': return 'move:'+r
    z=zhi
    while z>zlo-1e-9:
        r=d.move(lambda s,zz=z:lin_for(d,name,[x,y,zz]),step=0.02,tol=3e-3,grip=0.0,
                 stop_fn=lambda s:not grasping(s))
        d.grip(1.0)
        if not grasping(d.obs): return 'rel@%.4f'%z
        if r=='blocked': return 'blocked@%.4f'%z
        z-=dz
    return 'norelease'

def retreat(d):
    p=lp(d); return d.move(lambda s,u=p+np.array([0,0,0.14]):u,step=0.05,tol=0.01)

def place_low(d,name,x,y,carry=0.32,zstart=0.16,zmin=0.090,dz=0.0025):
    """carry to (x,y), descend with gripper CLOSED until blocked, then open."""
    pr=part_ref(d,name)
    r=d.move(lambda s:lin_for(d,name,[pr[0],pr[1],carry]),step=0.04,tol=5e-3)
    if r!='ok': return 'lift:'+r
    r=d.move(lambda s:lin_for(d,name,[x,y,carry]),step=0.04,tol=5e-3)
    if r!='ok': return 'move:'+r
    z=zstart; last=None
    while z>zmin:
        z-=dz
        r=d.move(lambda s,zz=z:lin_for(d,name,[x,y,zz]),step=0.02,tol=2e-3)
        cur=part_ref(d,name)[2]
        if r=='blocked' or cur>z+0.003: break
        last=cur
    zf=part_ref(d,name)[2]
    d.grip(1.0)
    if not grasping(d.obs): return 'rel@%.4f'%zf
    for k in range(6):
        d.grip(1.0)
        if not grasping(d.obs): return 'rel2@%.4f'%zf
    return 'norel@%.4f'%zf
