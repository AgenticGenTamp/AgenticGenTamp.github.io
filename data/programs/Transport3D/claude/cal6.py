import numpy as np, kin
from env_client import make_env
JN=["joint_%d"%i for i in range(1,8)]
MZ=0.245; FWD=0.12
def robot(o):
    R=o.get_object_from_name("robot")
    return (np.array([float(o.get(R,f)) for f in ["pos_base_x","pos_base_y","pos_base_rot"]]),
            np.array([float(o.get(R,j)) for j in JN]), float(o.get(R,"grasp_active")))
def opos(o,n):
    ob=o.get_object_from_name(n); return np.array([float(o.get(ob,f)) for f in ["pose_x","pose_y","pose_z"]])
def armbase(b): return (b[0]+FWD*np.cos(b[2]), b[1]+FWD*np.sin(b[2]))
env=make_env(); o,info=env.reset(seed=0)
cube=opos(o,"cube1")
ang=np.arctan2(cube[1],cube[0]); tgt=cube[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
for k in range(60):
    b,q,g=robot(o); e=np.concatenate([tgt-b[:2],[ang-b[2]]])
    if np.abs(e).max()<1e-3: break
    a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(e,-0.2,0.2); o,r,t,tr,info=env.step(a)
def goto(qd):
    global o
    for k in range(60):
        b,q,g=robot(o); e=qd-q
        if np.abs(e).max()<1e-4: return True
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(e,-0.2,0.2)
        qp=q.copy(); o,r,t,tr,info=env.step(a); b,q,g=robot(o)
        if np.allclose(q,qp): return False
    return False
b,q,g=robot(o); print("base",np.round(b,3),"cube",np.round(cube,3))
for z in [0.35,0.15,0.08,0.05,0.035,0.025,0.015]:
    b,q,g=robot(o); ax,ay=armbase(b)
    qd=kin.ik_top_down(np.array([cube[0],cube[1],z]),yaw=b[2],q_init=q,base_x=ax,base_y=ay,base_rot=b[2],mount=(0,0,MZ))
    if qd is None: print(z,"ikfail"); continue
    ok=goto(qd)
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0; o,r,t,tr,info=env.step(a)
    b,q2,g2=robot(o)
    print("z",z,"reach",ok,"grasp",g2,"cube",np.round(opos(o,"cube1"),3),
          "graspobj",float(o.get(o.get_object_from_name("cube1"),"grasp_active")))
    if g2>0.5:
        R=o.get_object_from_name("robot")
        print("grasp_tf",[round(float(o.get(R,f)),4) for f in ["grasp_tf_x","grasp_tf_y","grasp_tf_z","grasp_tf_qx","grasp_tf_qy","grasp_tf_qz","grasp_tf_qw"]])
        b,q2,g2=robot(o); ax,ay=armbase(b)
        qd=kin.ik_top_down(np.array([cube[0],cube[1],0.45]),yaw=b[2],q_init=q2,base_x=ax,base_y=ay,base_rot=b[2],mount=(0,0,MZ))
        print("lift ok",goto(qd),"cube now",np.round(opos(o,"cube1"),3))
        break
    a[10]=1.0; o,r,t,tr,info=env.step(a)
env.close()
