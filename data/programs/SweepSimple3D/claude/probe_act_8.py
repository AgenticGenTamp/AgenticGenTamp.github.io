from env_client import make_env
import numpy as np, time
POS=["pos_base_x","pos_base_y","pos_base_rot"]+[f"pos_arm_joint{i}" for i in range(1,8)]+["pos_gripper"]
out=open("/sandbox/out8.txt","w")
def W(*a): out.write(" ".join(str(x) for x in a)+"\n"); out.flush()
env=make_env(); t=time.time(); obs,_=env.reset(seed=0); W("reset %.0fs"%(time.time()-t))
R=env.observation_space.get_type("mujoco_tidybot_robot")
def rd():
    r=obs.get_objects(R)[0]; return np.array([float(obs.get(r,f)) for f in POS])
def step(a):
    global obs
    obs,_,_,_,_=env.step(np.array(a,dtype=np.float32)); return rd()
z=np.zeros(11)
for _ in range(5): p=step(z)
# linearity base x
for cmd in (0.01,0.03,0.1):
    a=np.zeros(11); a[0]=cmd
    for i in range(3): c=step(a); d=c[0]-p[0]; p=c
    W("basex cmd",cmd,"perstep %.5f ratio %.3f"%(d,d/cmd))
for cmd in (0.02,0.05,0.1):
    a=np.zeros(11); a[2]=cmd
    for i in range(3): c=step(a); d=c[2]-p[2]; p=c
    W("baserot cmd",cmd,"perstep %.5f ratio %.3f"%(d,d/cmd))
for _ in range(5): p=step(z)
# clean coupling: index5
a=np.zeros(11); a[5]=0.1
for i in range(5):
    c=step(a); W("idx5 step",i,"dj1..j4",np.round(c[3:7]-p[3:7],4).tolist()); p=c
for _ in range(6): p=step(z)
W("settle after idx5",np.round(p[3:7],4).tolist())
# gripper partial
a=np.zeros(11); a[10]=0.5
for i in range(4): c=step(a); W("grip0.5",i,round(float(c[10]),4)); p=c
a=np.zeros(11); a[10]=0.0
for i in range(2): p=step(a)
# joint1 limits (idx3)
a=np.zeros(11); a[3]=0.1
for i in range(200):
    c=step(a)
    if i%40==0 or i==199: W("j1+",i,round(float(c[3]),4))
p=c
a=np.zeros(11); a[3]=-0.1
for i in range(300):
    c=step(a)
    if i%50==0 or i==299: W("j1-",i,round(float(c[3]),4))
env.close(); out.close()
