import numpy as np, kin, sys
from env_client import make_env
JN=["joint_%d"%i for i in range(1,8)]
MZ=0.24
def robot(o):
    R=o.get_object_from_name("robot")
    return (np.array([float(o.get(R,f)) for f in ["pos_base_x","pos_base_y","pos_base_rot"]]),
            np.array([float(o.get(R,j)) for j in JN]), float(o.get(R,"grasp_active")))
def opos(o,n):
    ob=o.get_object_from_name(n); return np.array([float(o.get(ob,f)) for f in ["pose_x","pose_y","pose_z"]])
env=make_env(); o,info=env.reset(seed=0)
cube=opos(o,"cube1")
ang=np.arctan2(cube[1],cube[0]); tgt=cube[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
for k in range(60):
    b,q,g=robot(o); e=np.concatenate([tgt-b[:2],[ang-b[2]]])
    if np.abs(e).max()<1e-3: break
    a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(e,-0.2,0.2); o,r,t,tr,info=env.step(a)
b,q,g=robot(o)
def goto(qd):
    global o
    for k in range(60):
        b,q,g=robot(o); e=qd-q
        if np.abs(e).max()<1e-4: return True
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(e,-0.2,0.2)
        qp=q.copy(); o,r,t,tr,info=env.step(a); b,q,g=robot(o)
        if np.allclose(q,qp): return False
    return False
print("base",np.round(b,3),"cube",np.round(cube,3))
for z in [0.4,0.2,0.12,0.09,0.06,0.045,0.03,0.02,0.01]:
    b,q,g=robot(o)
    qd=kin.ik_top_down(np.array([cube[0],cube[1],z]),yaw=b[2],q_init=q,base_x=b[0],base_y=b[1],base_rot=b[2],mount=(0,0,MZ))
    if qd is None: print(z,"ikfail"); continue
    ok=goto(qd)
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0; o,r,t,tr,info=env.step(a)
    b,q2,g2=robot(o)
    print("z",z,"reach",ok,"grasp",g2,"cube",np.round(opos(o,"cube1"),3))
    if g2>0.5:
        # lift
        for k in range(5):
            b,q2,g2=robot(o)
            qd=kin.ik_top_down(np.array([cube[0],cube[1],0.5]),yaw=b[2],q_init=q2,base_x=b[0],base_y=b[1],base_rot=b[2],mount=(0,0,MZ))
            if qd is not None: goto(qd); break
        print("after lift cube", np.round(opos(o,"cube1"),3), "graspactive", robot(o)[2])
        break
    a[10]=1.0; o,r,t,tr,info=env.step(a)
env.close()
