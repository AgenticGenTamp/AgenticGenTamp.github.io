import numpy as np, kin
from env_client import make_env

JN=["joint_%d"%i for i in range(1,8)]
def robot(o):
    R=o.get_object_from_name("robot")
    q=np.array([float(o.get(R,j)) for j in JN])
    b=np.array([float(o.get(R,f)) for f in ["pos_base_x","pos_base_y","pos_base_rot"]])
    return b,q,float(o.get(R,"grasp_active")),float(o.get(R,"finger_state"))
def opos(o,n):
    ob=o.get_object_from_name(n)
    return np.array([float(o.get(ob,f)) for f in ["pose_x","pose_y","pose_z"]])

env=make_env(); o,info=env.reset(seed=0)
cube=opos(o,"cube1"); print("cube",cube)
# drive base
d=cube[:2]; ang=np.arctan2(d[1],d[0]); tgt=d-0.55*np.array([np.cos(ang),np.sin(ang)])
print("base target",tgt,"rot",ang)
for k in range(60):
    b,q,ga,fs=robot(o)
    err=np.concatenate([tgt-b[:2],[ang-b[2]]])
    if np.abs(err).max()<1e-3: break
    a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(err,-0.2,0.2)
    o,r,t,tr,info=env.step(a)
b,q,ga,fs=robot(o); print("base now",np.round(b,3))

def goto(q_des, nmax=40):
    global o
    for k in range(nmax):
        b,q,ga,fs=robot(o)
        e=q_des-q
        if np.abs(e).max()<1e-3: return True,q
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(e,-0.2,0.2)
        qprev=q.copy()
        o,r,t,tr,info=env.step(a)
        b,q2,ga,fs=robot(o)
        if np.allclose(q2,qprev): return False,q2
    return False,q

C=0.15   # guess mount_z - tool_z combined offset
b,q,ga,fs=robot(o)
for z in [0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.0,-0.05,-0.1]:
    tp=np.array([cube[0],cube[1],z])
    qd=kin.ik_top_down(tp, yaw=b[2], q_init=q, base_x=b[0],base_y=b[1],base_rot=b[2], mount=(0,0,C))
    if qd is None: print(z,"IK fail"); continue
    ok,q=goto(qd)
    b,qc,ga,fs=robot(o)
    err=np.abs(qc-qd).max()
    # close gripper
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0
    o,r,t,tr,info=env.step(a)
    b,qc,ga,fs=robot(o)
    print("z",z,"reached",ok,"qerr",round(float(err),3),"grasp",ga,"finger",fs, "cube",np.round(opos(o,"cube1"),3))
    a[10]=1.0; o,r,t,tr,info=env.step(a)
env.close()
