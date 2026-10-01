import numpy as np, fk
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=6)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
def J(): return np.array([g("robot","pos_arm_joint%d"%i) for i in range(1,8)])
def B(): return np.array([g("robot","pos_base_x"),g("robot","pos_base_y"),g("robot","pos_base_rot")])
def step(tgtq, base_d=(0,0,0), grip=0.0):
    global obs
    a=np.zeros(11,dtype=np.float32)
    a[0:3]=np.clip(base_d,-0.1,0.1)
    a[3:10]=np.clip(tgtq-J(),-0.1,0.1)
    a[10]=grip
    obs,rew,t,tr,i2=env.step(a); return rew,t
Rd=np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
q0=J()
print("cube",round(g("cube_0","x"),3),round(g("cube_0","y"),3),"base",np.round(B(),3))
for z in [0.30,0.15,0.05,0.0,-0.05,-0.10]:
    q=fk.ik(np.array([0.45,0.0,z]),Rd,q0,(0,0,0))
    for _ in range(70): step(q)
    err=np.max(np.abs(J()-q))
    print("z",z,"joint err",round(float(err),3), "q",np.round(q,2), "actual",np.round(J(),2))
    q0=J()
env.close()
