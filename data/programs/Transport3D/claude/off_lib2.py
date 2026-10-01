import numpy as np, kutil, kin
MZ=0.269
def fkq(c,q=None,b=None):
    if q is None: b,q,_=c.robot()
    ax,ay=c.armbase(b)
    return kin.fk(q,base_x=ax,base_y=ay,base_rot=b[2],mount=(0,0,MZ))
def vik(c,pos,yaw,tol=2e-3):
    """IK verified against fk; tries several seeds, returns (q, poserr) best."""
    b,q,_=c.robot(); ax,ay=c.armbase(b)
    kw=dict(base_x=ax,base_y=ay,base_rot=b[2],mount=(0,0,MZ))
    seeds=[q,kin.Q_HOME,kin.Q_RETRACT]
    rng=np.random.default_rng(0)
    lo,hi=kin.JOINT_LIMITS[:,0],kin.JOINT_LIMITS[:,1]
    seeds+= [np.clip(rng.uniform(np.maximum(lo,-3),np.minimum(hi,3)),lo,hi) for _ in range(6)]
    best=None;berr=1e9
    for s in seeds:
        qd=kin.ik_top_down(np.asarray(pos,float),yaw=yaw,q_init=s,**kw)
        T=kin.fk(qd,**kw)
        e=np.linalg.norm(T[:3,3]-np.asarray(pos,float))
        # orientation check: approach axis down
        e+=2.0*np.linalg.norm(T[:3,2]-np.array([0,0,-1.0]))
        if e<berr: berr=e; best=qd
        if e<tol: break
    return c.unwrap(best,q),berr
def vmove(c,pos,yaw,tol=2e-3):
    qd,e=vik(c,pos,yaw,tol)
    if e>tol: return False,e
    return c.goto(qd),e
def vpath(c,pos,yaw,step=0.05,tol=2e-3):
    T=fkq(c); cur=T[:3,3].copy(); pos=np.asarray(pos,float)
    n=max(1,int(np.ceil(np.linalg.norm(pos-cur)/step)))
    for i in range(1,n+1):
        ok,e=vmove(c,cur+(pos-cur)*i/n,yaw,tol)
        if not ok: return False
    return True
