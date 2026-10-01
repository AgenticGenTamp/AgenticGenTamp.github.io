import sys; sys.path.insert(0,'.')
from env_client import make_env
from pr2fk import *
import numpy as np
env = make_env()
T=env.observation_space.get_type
RF=env.observation_space.type_features[T("robot")]
def rstate(o):
    r=o.get_objects(T("robot"))[0]; return np.array([o.get(r,x) for x in RF])
def blocks(o): return o.get_objects(T("block"))
def bpose(o,b):
    return np.array([o.get(b,x) for x in ["pose_x","pose_y","pose_z","pose_qx","pose_qy","pose_qz","pose_qw","grasp_active"]])
def byaw(o,b):
    p=bpose(o,b); return 2*np.arctan2(p[5],p[6])
LAST={}
def step(o,a):
    o2,r,te,tr,info=env.step(np.asarray(a,dtype=np.float32)); LAST.update(r=r,te=te,tr=tr,info=info); return o2
def goto(o, base, q, grip=0.0, verbose=False):
    for _ in range(60):
        s=rstate(o)
        d=np.zeros(11,dtype=np.float32)
        d[0:3]=np.array(base)-s[0:3]; d[3:10]=np.array(q)-s[3:10]
        d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
        if np.abs(d[:10]).max()<1e-5: return o,False
        d[:10]=np.clip(d[:10],-0.2,0.2); d[10]=grip
        o2=step(o,d)
        if np.allclose(rstate(o2)[:10],s[:10],atol=1e-7):
            if verbose: print("REJ")
            return o2,True
        o=o2
    return o,False
def Rtop(yaw):
    xa=np.array([0,0,-1.]); ya=np.array([np.cos(yaw),np.sin(yaw),0]); za=np.cross(xa,ya)
    return np.stack([xa,ya,za],1)
def ikp(base,pos,yaw,q):
    q2,err=ik(base,np.array(pos,float),Rtop(yaw),q)
    return q2,err
def move_tool(o,base,pos,yaw,grip=0.0,steps=None):
    """move tool to pos in straight-ish line via waypoints; returns obs, rejected"""
    s=rstate(o); q=s[3:10]
    t0,_=fk(s[:3],q)
    n=steps or max(1,int(np.ceil(np.linalg.norm(np.array(pos)-t0)/0.01)))
    for i in range(1,n+1):
        p=t0+(np.array(pos)-t0)*i/n
        qn,err=ikp(base,p,yaw,q)
        if err>1e-3: print("ik err",err,p)
        o,rej=goto(o,base,qn,grip)
        if rej: return o,True
        q=qn
    return o,False
def tool(o):
    s=rstate(o); return fk(s[:3],s[3:10])
def grip(o,g,n=1):
    for _ in range(n):
        a=np.zeros(11,np.float32); a[10]=g; o=step(o,a)
    return o
BASE=(-0.6,0.0,0.0)
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def pick(o,b,base=BASE,z=0.80,dyaw=0.0,lift=0.95):
    p=bpose(o,b); yaw=byaw(o,b)
    yaw=yaw-np.round(yaw/(np.pi/2))*(np.pi/2)+dyaw   # pick equivalent yaw near 0
    o,r1=move_tool(o,base,[p[0],p[1],lift],yaw); o,r2=move_tool(o,base,[p[0],p[1],z],yaw); o=grip(o,-1)
    ok=rstate(o)[11]>0.5
    o,r3=move_tool(o,base,[p[0],p[1],lift],yaw)
    return o,ok,yaw,(r1,r2,r3)
def carry(o,b,xy,block_yaw,z=0.95,base=BASE):
    # held: rotate tool so that block yaw becomes block_yaw
    s=rstate(o); t,R=tool(o); tyaw=np.arctan2(R[1,1],R[0,1])
    ty=tyaw+wrap(block_yaw-byaw(o,b))
    o,rej=move_tool(o,base,[xy[0],xy[1],z],ty); return o,rej
_rng=np.random.default_rng(0)
def ikp(base,pos,yaw,q):
    q2,err=ik(base,np.array(pos,float),Rtop(yaw),q)
    if err<1e-4: return q2,err
    best=(err,q2)
    for _ in range(30):
        qi=_rng.uniform(LO,HI); qs,e=ik(base,np.array(pos,float),Rtop(yaw),qi)
        if e<1e-4:
            qs[[2,4,6]]=q[[2,4,6]]+wrap(qs[[2,4,6]]-q[[2,4,6]])
            qs=np.clip(qs,LO,HI)
            c=np.linalg.norm(qs-q)
            if best[0]>1e-4 or c<np.linalg.norm(best[1]-q): best=(e,qs)
    return best[1],best[0]
def best_yaw(o,base,pos,yaw0):
    q=rstate(o)[3:10]; best=None
    for k in range(4):
        y=wrap(yaw0+k*np.pi/2)
        qs,e=ikp(base,pos,y,q)
        marg=np.min(np.minimum(qs-LO,HI-qs))
        sc=e*1000+np.linalg.norm(qs-q)+(5 if marg<0.4 else 0)
        if best is None or sc<best[0]: best=(sc,y)
    return best[1]
def pick(o,b,base=BASE,z=0.80,dyaw=0.0,lift=0.95):
    p=bpose(o,b)
    yaw=best_yaw(o,base,[p[0],p[1],z],byaw(o,b))+dyaw
    o,r1=move_tool(o,base,[p[0],p[1],lift],yaw); o,r2=move_tool(o,base,[p[0],p[1],z],yaw); o=grip(o,-1)
    ok=rstate(o)[11]>0.5
    o,r3=move_tool(o,base,[p[0],p[1],lift],yaw)
    return o,ok,yaw,(r1,r2,r3)
def carry(o,b,xy,block_yaw,z=0.95,base=BASE):
    t,R=tool(o); tyaw=np.arctan2(R[1,1],R[0,1])
    ty=tyaw+wrap(block_yaw-byaw(o,b))
    ty=best_yaw(o,base,[xy[0],xy[1],z],ty)
    o,rej=move_tool(o,base,[xy[0],xy[1],z],ty); return o,rej
