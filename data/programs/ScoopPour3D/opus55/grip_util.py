import numpy as np, kin, time
from calib_util import R, RD
def cyaw(qq):
    w,x,y,z=qq; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
class G(R):
    def __init__(s, seed=0):
        super().__init__(seed); s.I=np.zeros(7); s.ki=0.3; s.settle_v=0.003; s.qs=s.q().copy()
    def cubes(s):
        return {n:s.P(n) for n in s.obs.get_object_names() if n.startswith('cube')}
    def bins(s):
        out={}
        for n in s.obs.get_object_names():
            if n.startswith('bin'):
                o=s.obs.get_object_from_name(n); out[n]=np.array([s.obs.get(o,k) for k in ('x','y','z','bb_x','bb_y','bb_z')])
        return out
    def ctrl(s, pw, Rw, bt, integ=True, tol=0.003, maxsteps=80, watch=None):
        """drive base to bt and tool to pw; returns steps used"""
        for k in range(maxsteps):
            b=s.base(); d=np.array(bt,float)-b; d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
            db=np.array([d[0]/.87,d[1]/.87,d[2]/.99])
            pa=kin.world_to_arm(pw,np.array(bt,float)); Ra=kin.rotz(-bt[2])@Rw
            qt,_=kin.ik_arm(pa,Ra,s.qs,iters=100,lam=0.05); s.qs=qt
            q=s.q(); qp=getattr(s,'qprev',q); s.qprev=q
            if integ and np.max(np.abs(qt+s.I-s.qi))<0.05 and np.max(np.abs(q-qp))<s.settle_v:
                s.I=np.clip(s.I+s.ki*(qt-q),-0.2,0.2)
            qc=qt+s.I; dq=np.clip((qc-s.qi)/0.25,-0.1,0.1)
            s.step(db=db,dq=dq)
            if watch: watch(s)
            p,_=kin.fk_world(s.base(),s.q()); e=np.linalg.norm(p-pw)
            s.trace.append(e) if hasattr(s,'trace') else None
            if np.max(np.abs(qc-s.qi))<1e-3 and e<tol and np.all(np.abs(d[:2])<0.004): return k
        return maxsteps
    def hold(s, pw, Rw, bt, n):
        for k in range(n):
            b=s.base(); d=np.array(bt,float)-b
            qc=s.qs+s.I; dq=np.clip((qc-s.qi)/0.25,-0.1,0.1)
            s.step(db=[d[0]/.87,d[1]/.87,0],dq=dq)
    def tool(s):
        return kin.fk_world(s.base(),s.q())[0]
def isolated(cubes, name, r=0.04):
    p=cubes[name]; return all(np.linalg.norm(cubes[m][:2]-p[:2])>=r for m in cubes if m!=name)

HZ=0.56
def gyaw(g,name,off=0.0):
    y=cyaw(g.Q(name)); y=(y+np.pi/4)%(np.pi/2)-np.pi/4
    return y+off
def pick(g, name, open_cmd=0.0, dz=-0.017, dy=0.0, dx=0.0, yaw_off=0.0, nclose=6, verbose=False):
    c0=g.P(name).copy(); yw=gyaw(g,name,yaw_off); Rw=kin.rotz(yw)@RD
    # target in gripper frame offsets: dy along closing axis (rotated), dx perpendicular
    cs,sn=np.cos(yw),np.sin(yw)
    off=np.array([cs*dx-sn*dy, sn*dx+cs*dy, 0])
    bt=np.array([-0.15,c0[1],0.])
    g.s_g=open_cmd; g.g=open_cmd; g.I[:]=0
    g.ctrl(c0+off+[0,0,HZ-c0[2]],Rw,bt,tol=0.004,maxsteps=150)
    g.ctrl(c0+off+[0,0,0.02+dz],Rw,bt,tol=0.003,maxsteps=80)
    tgt=c0+off+[0,0,dz]
    g.ctrl(tgt,Rw,bt,tol=0.002,maxsteps=50)
    tool_pre=g.tool(); cpre=g.P(name).copy()
    g.g=1.0; g.hold(tgt,Rw,bt,nclose)
    cclose=g.P(name).copy(); tool_post=g.tool()
    g.ctrl(c0+off+[0,0,HZ-c0[2]],Rw,bt,tol=0.02,maxsteps=60)
    c1=g.P(name); tl=g.tool()
    ok=c1[2]>0.52
    info=dict(ok=ok, tool_err=(tool_pre-tgt).round(4), push=np.linalg.norm(cpre[:2]-c0[:2]).round(4),
              pushz=round(cpre[2]-c0[2],4), rise=(tool_post-tool_pre).round(4), cmove=(cclose-cpre).round(4), held_off=(c1-tl).round(4), cz=round(c1[2],4))
    if verbose: print(name, info, flush=True)
    return ok, info
def place(g, name, xy, rel=0.004):
    """carry held cube to xy and release with bottom ~rel above floor"""
    yw=gyaw(g,name); Rw=kin.rotz(yw)@RD
    c=g.P(name); tl=g.tool(); ho=c-tl
    bt=np.array([-0.15,xy[1],0.])
    g.ctrl(np.array([xy[0],xy[1],HZ]),Rw,bt,tol=0.01,maxsteps=150)
    tz=0.4825+rel-ho[2]
    g.ctrl(np.array([xy[0]-ho[0],xy[1]-ho[1],tz]),Rw,bt,tol=0.003,maxsteps=80)
    g.g=0.0; g.hold(None,Rw,bt,8)
    g.ctrl(np.array([xy[0],xy[1],HZ]),Rw,bt,tol=0.02,maxsteps=60)
    return g.P(name)
