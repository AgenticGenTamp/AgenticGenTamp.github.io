import numpy as np
from env_client import make_env
from probe_fk_world import servo2, ee_world
np.set_printoptions(precision=4,suppress=True,linewidth=250)
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
C=o[:80].reshape(5,16)[:,:3].copy(); tot=0
def rep(t,o,e,u):
    global tot; tot+=u; c=o[:80].reshape(5,16)[:,:3]
    print(f"{t:14s} ee={np.round(ee_world(o),4)} perr={e:.4f} u={u} tot={tot} cubes_moved={np.round(np.linalg.norm(c-C,axis=1),3)}")
    return c
# accuracy re-check with floor
for p in [[0.75,-0.15,0.62],[0.85,-0.35,0.60],[0.70,-0.05,0.65]]:
    o,e,u=servo2(env,o,p,steps=200,grip=0.0); rep(f"free{p[1]}",o,e,u)
# pick cube3 place at (0.82,-0.30)
t=C[3]
o,e,u=servo2(env,o,[t[0],t[1],0.56],steps=150,grip=0.0); rep("above",o,e,u)
o,e,u=servo2(env,o,[t[0],t[1],0.470],steps=150,grip=0.0); rep("descend",o,e,u)
for _ in range(15):
    a=np.zeros(11); a[10]=1.0; o,_,_,_,_=env.step(a); o=np.asarray(o,float)
tot+=15
o,e,u=servo2(env,o,[t[0],t[1],0.62],steps=120,grip=1.0); rep("lift",o,e,u)
GOAL=[0.82,-0.30]
o,e,u=servo2(env,o,[GOAL[0],GOAL[1],0.62],steps=200,grip=1.0); rep("carry",o,e,u)
o,e,u=servo2(env,o,[GOAL[0],GOAL[1],0.478],steps=150,grip=1.0); rep("place",o,e,u)
print("  cube3 while held:",np.round(o[:80].reshape(5,16)[3,:3],4),"ee",np.round(ee_world(o),4))
for _ in range(10):
    a=np.zeros(11); a[10]=0.0; o,_,_,_,_=env.step(a); o=np.asarray(o,float)
o,e,u=servo2(env,o,[GOAL[0],GOAL[1],0.62],steps=100,grip=0.0); rep("retreat",o,e,u)
print("FINAL cube3:",np.round(o[:80].reshape(5,16)[3,:3],4),"goal",GOAL)
env.close()
