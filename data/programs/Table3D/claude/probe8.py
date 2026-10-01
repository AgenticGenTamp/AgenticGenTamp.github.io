import numpy as np
from env_client import make_env
from fk import fk
from ik2 import solve_pos_axis
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
def run(seed=0,cube="cube0",yaw=0.0):
    env=make_env(); obs,_=env.reset(seed=seed)
    c=obs.get_object_from_name(cube)
    cp=np.array([float(obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
    q=getq(obs)
    def goto(qt,lim=0.2):
        nonlocal q,obs
        rej=0
        for _ in range(100):
            d=qt-q
            if np.max(np.abs(d))<2e-3: return True
            a=np.zeros(11); a[3:10]=np.clip(d,-lim,lim)
            o2,*_=env.step(a); qn=getq(o2)
            if np.allclose(qn,q,atol=1e-8):
                rej+=1
                if rej>2: return False
            q=qn; obs=o2
        return False
    # open gripper first
    a=np.zeros(11); a[10]=1.0; obs,*_=env.step(a)
    dz=0.30
    qt,pe,ae=solve_pos_axis(cp+np.array([0,0,dz]),0.0)
    if not goto(qt): print("cant reach start",pe,ae); env.close(); return
    while dz>-0.10:
        # try close
        a=np.zeros(11); a[10]=-1.0; o2,rw,t,tr,i=env.step(a)
        r=o2.get_object_from_name("robot"); ga=float(o2.get(r,"grasp_active")); fs=float(o2.get(r,"finger_state"))
        if ga>0.5 or fs!=0.0:
            print("GRASP at dz=%.3f ga=%s fs=%s term=%s"%(dz,ga,fs,t)); env.close(); return dz
        a=np.zeros(11); a[10]=1.0; obs,*_=env.step(a)
        dz-=0.01
        qn,pe,ae=solve_pos_axis(cp+np.array([0,0,dz]),0.0,q0=q)
        if pe>0.004: print("ik fail dz",round(dz,3)); break
        if not goto(qn):
            print("BLOCKED at dz=%.3f"%dz); break
    print("no grasp, last dz=%.3f"%dz); env.close()
run()
