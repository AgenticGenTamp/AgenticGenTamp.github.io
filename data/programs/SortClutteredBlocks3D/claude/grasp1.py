import numpy as np, sys
from env_client import make_env
from ctrl import robot_state, action, cubes
from fk import ik, fk
TH=np.pi
def RD(yaw=0.0):
    c,s=np.cos(yaw),np.sin(yaw)
    Rz=np.array([[c,-s,0],[s,c,0],[0,0,1.0]])
    return Rz@np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
MZ=0.386
def run(mx,my,mz,gh,yaw,seed=0,verbose=False):
    env=make_env(); obs,info=env.reset(seed=seed)
    b,q,g=robot_state(obs)
    cb=cubes(obs)
    # pick most isolated cube
    names=list(cb); 
    best=max(names, key=lambda n: min(np.linalg.norm(cb[n][:2]-cb[m][:2]) for m in names if m!=n))
    tgt=cb[best]
    def world_to_base(w, arm_p):
        # world = base + R(TH)@(mount+arm_p_xy) ; R(TH)=-I for TH=pi
        return np.array([w[0]+ (mx+arm_p[0]), w[1] + (my+arm_p[1]), TH])
    def step_to(arm_p, bt, n, grip):
        nonlocal b,q,g,obs
        qt=ik(np.array(arm_p), RD(yaw), q)
        for t in range(n):
            a=action(b,q,bt,qt,grip); obs,r,te,tr,i=env.step(a); b,q,g=robot_state(obs)
        return fk(q)[:3,3]
    R=0.45
    app=np.array([R,0.0,0.12])
    bt=world_to_base(tgt, app)
    p=step_to(app, bt, 140, 0.0)
    down=np.array([R,0.0,gh])
    p2=step_to(down, bt, 40, 0.0)
    step_to(down, bt, 3, 1.0)
    p3=step_to(app, bt, 40, 1.0)
    cb2=cubes(obs)
    res = {k: np.round(cb2[k]-cb[k],3) for k in cb}
    env.close()
    return best, tgt, p, p2, p3, res
for mz in [0.386]:
  for gh in [0.41-mz, 0.41-mz-0.01]:
    print('gh',round(gh,3), run(0,0,mz,gh,0.0))
