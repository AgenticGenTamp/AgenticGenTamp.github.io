import numpy as np, kin
from env_client import make_env
JN=["joint_%d"%i for i in range(1,8)]
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
    b,q,ga=robot(o); e=np.concatenate([tgt-b[:2],[ang-b[2]]])
    if np.abs(e).max()<1e-3: break
    a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(e,-0.2,0.2); o,r,t,tr,info=env.step(a)
b,q,ga=robot(o); print("base",np.round(b,3),"cube",np.round(cube,3))
def goto(qd,tag=""):
    global o
    for k in range(60):
        b,q,ga=robot(o); e=qd-q
        if np.abs(e).max()<1e-4: return True
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(e,-0.2,0.2)
        qp=q.copy(); o,r,t,tr,info=env.step(a); b,q,ga=robot(o)
        if np.allclose(q,qp): return False
    return False
z=0.7
qd=kin.ik_top_down(np.array([cube[0],cube[1],z]), yaw=b[2], q_init=q, base_x=b[0],base_y=b[1],base_rot=b[2])
print("ik ok", qd is not None)
if qd is not None:
    print("model EE at qd", np.round(kin.fk(qd,base_x=b[0],base_y=b[1],base_rot=b[2])[:3,3],4))
    ok=goto(qd); b,q,ga=robot(o)
    print("reached",ok,"actual model EE",np.round(kin.fk(q,base_x=b[0],base_y=b[1],base_rot=b[2])[:3,3],4))
prev_ok=True
while z>-0.7:
    z-=0.02
    b,q,ga=robot(o)
    qd=kin.ik_top_down(np.array([cube[0],cube[1],z]), yaw=b[2], q_init=q, base_x=b[0],base_y=b[1],base_rot=b[2])
    if qd is None: print("z",round(z,3),"IK FAIL"); continue
    ok=goto(qd); b,q2,ga=robot(o)
    ee=kin.fk(q2,base_x=b[0],base_y=b[1],base_rot=b[2])[:3,3]
    # try grasp
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0; o,r,t,tr,info=env.step(a)
    b,q3,ga2=robot(o)
    cn=opos(o,"cube1")
    print("z",round(z,3),"reach",ok,"eeZ",round(float(ee[2]),3),"grasp",ga2,"cubez",round(float(cn[2]),3))
    a[10]=1.0; o,r,t,tr,info=env.step(a)
    if not ok: print("BLOCKED at z",round(z,3)); break
    if ga2>0.5: print("GRASPED at z",round(z,3)); break
env.close()
