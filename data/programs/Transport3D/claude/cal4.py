import numpy as np, kin
from env_client import make_env
JN=["joint_%d"%i for i in range(1,8)]
def robot(o):
    R=o.get_object_from_name("robot")
    return (np.array([float(o.get(R,f)) for f in ["pos_base_x","pos_base_y","pos_base_rot"]]),
            np.array([float(o.get(R,j)) for j in JN]))
env=make_env(); o,info=env.reset(seed=1)
b,q=robot(o)
def goto(qd):
    global o
    for k in range(60):
        b,q=robot(o); e=qd-q
        if np.abs(e).max()<1e-4: return True
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(e,-0.2,0.2)
        qp=q.copy(); o,r,t,tr,info=env.step(a); b,q=robot(o)
        if np.allclose(q,qp): return False
    return False
for (px,py) in [(0.6,0.0),(0.5,0.2),(0.45,-0.1),(0.7,0.0)]:
    # go high first
    b,q=robot(o)
    qd=kin.ik_top_down(np.array([px,py,0.95]),yaw=0.0,q_init=q,base_x=b[0],base_y=b[1],base_rot=b[2])
    if qd is None: print(px,py,"high IK fail"); continue
    if not goto(qd): print(px,py,"cannot reach high"); continue
    zb=None
    z=0.95
    while z>0.0:
        z-=0.02
        b,q=robot(o)
        qd=kin.ik_top_down(np.array([px,py,z]),yaw=0.0,q_init=q,base_x=b[0],base_y=b[1],base_rot=b[2])
        if qd is None: print("  ikfail at",round(z,2)); continue
        if not goto(qd):
            zb=z; break
    b,q=robot(o)
    print("probe",(px,py),"blocked model z=",None if zb is None else round(zb,3),"final ee",np.round(kin.fk(q,base_x=b[0],base_y=b[1],base_rot=b[2])[:3,3],3))
env.close()
