import numpy as np, sys
from env_client import make_env
from fk import fk
from ik2 import solve_pos_axis
J=["joint_%d"%i for i in range(1,8)]
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
def run(tool_len, base_z=0.0, seed=0, cube="cube0"):
    env=make_env(); obs,_=env.reset(seed=seed)
    c=obs.get_object_from_name(cube)
    cp=np.array([float(obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]]) - np.array([0,0,base_z])
    q=getq(obs)
    qpre,pe1,ae1=solve_pos_axis(cp+np.array([0,0,0.12]),tool_len)
    qg,pe2,ae2=solve_pos_axis(cp,tool_len,q0=qpre)
    res=[]
    def goto(qt):
        nonlocal q,obs
        rej=0
        for _ in range(80):
            d=qt-q
            if np.max(np.abs(d))<2e-3: return True
            a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2)
            o2,*_=env.step(a); qn=getq(o2)
            if np.allclose(qn,q,atol=1e-8):
                rej+=1
                if rej>2: return False
            q=qn; obs=o2
        return False
    ok1=goto(qpre); ok2=goto(qg)
    a=np.zeros(11); a[10]=-1.0; obs,*_=env.step(a)
    r=obs.get_object_from_name("robot")
    ga=float(obs.get(r,"grasp_active"))
    print("tool %.2f basez %.2f ikerr %.3f/%.3f %.3f/%.3f pre %s grasp %s -> grasp_active %s"%(
        tool_len,base_z,pe1,ae1,pe2,ae2,ok1,ok2,ga))
    env.close(); return ga
for bz in [0.0,0.4]:
    for tl in [0.0,0.05,0.10,0.15,0.20,0.25]:
        run(tl,bz)
