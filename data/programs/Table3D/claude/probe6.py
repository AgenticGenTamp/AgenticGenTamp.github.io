import numpy as np
from env_client import make_env
from fk import fk
from ik import solve, ik
J=["joint_%d"%i for i in range(1,8)]
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
def T_of(pos,yaw=0.0):
    Rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
    Rd=np.array([[1,0,0],[0,-1,0],[0,0,-1.]])
    T=np.eye(4); T[:3,:3]=Rz@Rd; T[:3,3]=pos; return T

def descend(x,y,z0=0.55,zmin=0.0,dz=0.02,seed=0):
    env=make_env(); obs,_=env.reset(seed=seed)
    q=getq(obs)
    qt,pe,oe=solve(T_of([x,y,z0]),0.0)
    def goto(qt):
        nonlocal q,obs
        for _ in range(60):
            d=qt-q
            if np.max(np.abs(d))<2e-3: return True
            a=np.zeros(11); a[3:10]=np.clip(d,-0.25,0.25)
            o2,*_=env.step(a); qn=getq(o2)
            if np.allclose(qn,q,atol=1e-8): return False
            q=qn; obs=o2
        return False
    if not goto(qt): print("cant reach start"); env.close(); return None
    z=z0
    while z>zmin:
        z-=dz
        qn,pe,oe=ik(T_of([x,y,z]),q,0.0)
        if pe>0.005: print("  ik fail at",round(z,3)); break
        if not goto(qn):
            print("  BLOCKED going to z=%.3f (last ok fk_z=%.3f)"%(z,fk(q)[2,3])); env.close(); return fk(q)[2,3]
    print("  reached bottom fk_z=%.3f"%fk(q)[2,3])
    env.close(); return fk(q)[2,3]

print("over table empty (0.55,-0.35)"); descend(0.55,-0.35)
print("over floor (0.15,0.0)"); descend(0.15,0.0)
print("above cube0 (0.648,-0.173)"); descend(0.648,-0.173)
