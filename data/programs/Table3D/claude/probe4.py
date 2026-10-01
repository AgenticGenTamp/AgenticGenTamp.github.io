import numpy as np
from env_client import make_env
from fk import fk
from ik import solve
J=["joint_%d"%i for i in range(1,8)]
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
def target_T(pos,yaw=0.0):
    Rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
    Rd=np.array([[1,0,0],[0,-1,0],[0,0,-1.]])
    T=np.eye(4); T[:3,:3]=Rz@Rd; T[:3,3]=pos; return T
# descend over empty table spot (0.55, -0.35) far from cubes
for z in [0.7,0.6,0.55,0.5,0.45,0.42,0.40,0.35,0.30,0.2]:
    env=make_env(); obs,_=env.reset(seed=0)
    q=getq(obs)
    qt,pe,oe=solve(target_T(np.array([0.55,-0.35,z])),0.0)
    if pe>0.02: print("z",z,"IK FAIL",round(pe,3)); env.close(); continue
    rejected=0; steps=0
    for _ in range(80):
        d=qt-q
        if np.max(np.abs(d))<1e-3: break
        a=np.zeros(11); a[3:10]=np.clip(d,-0.3,0.3)
        o2,r,t,tr,i=env.step(a); qn=getq(o2); steps+=1
        if np.allclose(qn,q,atol=1e-7): rejected+=1
        if rejected>3: break
        q=qn
    print("z",z,"reached", np.max(np.abs(qt-q))<1e-2, "rejects",rejected, "fk_z", round(fk(q)[2,3],3))
    env.close()
