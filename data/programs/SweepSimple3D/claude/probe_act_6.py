from env_client import make_env
import numpy as np, time
POS=["pos_base_x","pos_base_y","pos_base_rot"]+[f"pos_arm_joint{i}" for i in range(1,8)]+["pos_gripper"]
out=open("/sandbox/out6.txt","w")
def W(*a):
    out.write(" ".join(str(x) for x in a)+"\n"); out.flush()
env=make_env(); t=time.time()
obs,_=env.reset(seed=0)
W("reset %.0fs"%(time.time()-t))
R=env.observation_space.get_type("mujoco_tidybot_robot")
def rd(o):
    r=o.get_objects(R)[0]
    return np.array([float(o.get(r,f)) for f in POS])
def step(a):
    global obs
    obs,_,_,_,_=env.step(np.array(a,dtype=np.float32))
    return rd(obs)
W("INIT",np.round(rd(obs),4).tolist())
# per-index mapping
for idx in range(10):
    a=np.zeros(11); a[idx]=0.1
    p1=step(a); p2=step(a); p3=step(a)
    d=p3-p2
    W("act[%d]=+0.1 perstep"%idx,[(POS[j],round(float(d[j]),4)) for j in range(11) if abs(d[j])>5e-4])
    a[idx]=-0.1
    for _ in range(3): step(a)
    W("   after undo, resid vs init:",[(POS[j],round(float(rd(obs)[j]),4)) for j in range(11)])
env.close(); out.close()
