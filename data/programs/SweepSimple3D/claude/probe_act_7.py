from env_client import make_env
import numpy as np, time
POS=["pos_base_x","pos_base_y","pos_base_rot"]+[f"pos_arm_joint{i}" for i in range(1,8)]+["pos_gripper"]
out=open("/sandbox/out7.txt","w")
def W(*a): out.write(" ".join(str(x) for x in a)+"\n"); out.flush()
env=make_env(); t=time.time(); obs,_=env.reset(seed=0); W("reset %.0fs"%(time.time()-t))
R=env.observation_space.get_type("mujoco_tidybot_robot")
def rd():
    r=obs.get_objects(R)[0]; return np.array([float(obs.get(r,f)) for f in POS])
def step(a):
    global obs
    obs,_,_,_,_=env.step(np.array(a,dtype=np.float32)); return rd()
def s(v): return np.round(v[:3],4).tolist()
p=rd(); W("INIT",s(p))
a=np.zeros(11); a[1]=0.1
for i in range(6): c=step(a); W("y+",i,s(c),"d",np.round(c[:3]-p[:3],4).tolist()); p=c
a[1]=-0.1
for i in range(6): c=step(a); W("y-",i,s(c),"d",np.round(c[:3]-p[:3],4).tolist()); p=c
W("--- rotate then move x ---")
a=np.zeros(11); a[2]=0.1
for i in range(9): c=step(a); p=c
W("after rot",s(p))
a=np.zeros(11); a[0]=0.1
for i in range(3): c=step(a); W("x+ afterrot d",np.round(c[:3]-p[:3],4).tolist()); p=c
a=np.zeros(11); a[1]=0.1
for i in range(3): c=step(a); W("y+ afterrot d",np.round(c[:3]-p[:3],4).tolist()); p=c
W("--- gripper ---")
a=np.zeros(11); a[10]=1.0
for i in range(14):
    c=step(a)
    if i%2==0: W("close",i,round(float(c[10]),4))
a[10]=0.0
for i in range(14):
    c=step(a)
    if i%2==0: W("open",i,round(float(c[10]),4))
W("--- joint limit idx4 (joint2) ---")
a=np.zeros(11); a[4]=0.1
for i in range(60):
    c=step(a)
    if i%10==0 or i==59: W("j2",i,round(float(c[4]),4))
a=np.zeros(11); a[4]=-0.1
for i in range(80):
    c=step(a)
    if i%20==0 or i==79: W("j2neg",i,round(float(c[4]),4))
env.close(); out.close()
