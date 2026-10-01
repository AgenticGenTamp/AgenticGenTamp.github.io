import numpy as np
from env_client import make_env

def f(s,o,k): return float(s.get(o,k))

def run(seed, startx, diagx, diagy, theta_action=0.):
    env=make_env(); s,info=env.reset(seed=seed); typ=env.observation_space.get_type("kin_robot"); r=s.get_objects(typ)[0]
    small=[s.get_object_from_name(n) for n in s.get_object_names() if n.startswith("small")]
    print("seed",seed,"n",len(small),"r0",*[round(f(s,r,k),3) for k in ("x","y","theta","base_radius","arm_length","arm_joint","gripper_base_width","finger_gap")])
    targets=[(startx,2.15,50),(startx,.3,80),(diagx,diagy,100),(2.4,diagy,50)]
    term=False
    for tx,ty,n in targets:
      for _ in range(n):
        x,y=f(s,r,"x"),f(s,r,"y")
        a=np.array([np.clip(tx-x,-.03,.03),np.clip(ty-y,-.03,.03),theta_action,-.08,0],np.float32)
        s,re,term,tr,inf=env.step(a)
        if term or tr: break
      vals=[(f(s,o,"x"),f(s,o,"y")) for o in small]
      print(" target",tx,ty,"rob",round(f(s,r,"x"),2),round(f(s,r,"y"),2),"right",sum(x>1.75 for x,y in vals),"high",sum(y>1.4 for x,y in vals),"max",tuple(round(z,2) for z in max(vals,key=lambda z:z[0])),"term",term)
      if term or tr: break
    env.close()

for i,(sx,dx,dy,da) in enumerate([(1.25,2.3,2.0,0),(1.4,2.3,2.,0),(.9,2.3,2.,0),(1.3,2.3,1.7,.098)]): run(210+i,sx,dx,dy,da)
