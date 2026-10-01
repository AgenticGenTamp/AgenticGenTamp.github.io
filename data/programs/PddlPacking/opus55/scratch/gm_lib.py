from env_client import make_env
import numpy as np, math
from kin import fk, ik_down, Q0, wrap, CONT
env=make_env(); T=env.observation_space.get_type
f=list(env.observation_space.type_features[T("robot")])
BASE=None; NSTEP=[0]
def rstate(o):
    r=o.get_objects(T("robot"))[0]; return np.array([o.get(r,x) for x in f],float)
def block(o):
    b=o.get_objects(T("block"))[0]; g=lambda k: float(o.get(b,k))
    return np.array([g("pose_x"),g("pose_y"),g("pose_z")]), 2*math.atan2(g("pose_qz"),g("pose_qw"))
class R:
    def __init__(s,seed,base,grip=None):
        s.o,_=env.reset(seed=seed); s.base=np.array(base)
        s.goto_base(base)
        s.q=rstate(s.o)[3:10]
        if grip is not None: s.grip(grip)
    def step(s,d10,g=0.0):
        a=np.zeros(11,np.float32); a[:10]=d10; a[10]=g
        before=rstate(s.o)
        o2,*_=env.step(a); NSTEP[0]+=1
        after=rstate(o2); s.o=o2
        ok=np.abs(after[:10]-before[:10]).max()>1e-7 or np.abs(d10).max()<1e-9
        return ok
    def goto_base(s,b):
        for _ in range(100):
            c=rstate(s.o)[:3]; d=np.zeros(10); d[:3]=np.clip(np.array(b)-c,-0.2,0.2)
            if np.abs(d).max()<1e-7: return
            assert s.step(d),"base fail"
    def grip(s,g):
        for _ in range(5): s.step(np.zeros(10),g)
        return rstate(s.o)[10]
    def cfg(s): return rstate(s.o)[3:10]
    def move_q(s,qt,maxd=0.1):
        qc=s.cfg(); d=np.array(qt)-qc; d=np.where(CONT,wrap(d),d)
        n=max(1,int(math.ceil(np.abs(d).max()/maxd)))
        for i in range(n):
            if not s.step(np.r_[0,0,0,d/n]): return False
        return True
    def ik(s,pos,yaw):
        cur=s.cfg(); best=None
        seeds=[cur,Q0]+[Q0+np.random.RandomState(i).uniform(-0.6,0.6,7) for i in range(12)]
        for sd in seeds:
            q,ok=ik_down(s.base,np.array(pos,float),sd,yaw=yaw)
            if ok:
                d=q-cur; d=np.where(CONT,wrap(d),d); c=np.abs(d).max()
                if best is None or c<best[0]: best=(c,q)
                if c<0.05: break
        if best is None: return None,False
        return best[1],True
    def move_tool(s,pos,yaw,maxd=0.1):
        q,ok=s.ik(pos,yaw); assert ok,("ik fail",pos)
        return s.move_q(q,maxd)
    def try_tool(s,pos,yaw):
        q,ok=s.ik(pos,yaw); assert ok,("ik fail",pos)
        return s.move_q(q,0.2)
    def sweep(s,p_far,p_near,yaw,steps=(0.01,0.002,0.0005,0.0002)):
        """move from p_far toward p_near; return largest param t (dist from p_near, along line) still free"""
        p_far=np.array(p_far,float); p_near=np.array(p_near,float)
        L=np.linalg.norm(p_far-p_near); u=(p_far-p_near)/L
        # go high above far, then descend to far
        hi=p_far.copy(); hi[2]=max(1.0,p_far[2])
        assert s.move_tool(hi,yaw),"hi fail"
        assert s.move_tool(p_far,yaw),"far fail"
        t=L
        for st in steps:
            while t-st>=-1e-9:
                if s.try_tool(p_near+u*(t-st),yaw): t-=st
                else: break
            if t<=1e-9: return 0.0
        return t
