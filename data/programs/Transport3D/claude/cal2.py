import numpy as np, kin
from env_client import make_env
JN=["joint_%d"%i for i in range(1,8)]
def robot(o):
    R=o.get_object_from_name("robot")
    return (np.array([float(o.get(R,f)) for f in ["pos_base_x","pos_base_y","pos_base_rot"]]),
            np.array([float(o.get(R,j)) for j in JN]))
env=make_env(); o,info=env.reset(seed=1)
# drive base to (0,-2) free space
for k in range(40):
    b,q=robot(o)
    e=np.array([0.0,-2.0])-b[:2]
    if np.abs(e).max()<1e-3: break
    a=np.zeros(11,dtype=np.float32); a[:2]=np.clip(e,-0.2,0.2); o,r,t,tr,info=env.step(a)
b,q=robot(o); print("base",np.round(b,3))
for ji in range(7):
    for sign in [1,-1]:
        # reset arm to retract
        qr=np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
        for k in range(40):
            b,q=robot(o); e=qr-q
            if np.abs(e).max()<1e-3: break
            a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(e,-0.2,0.2); o,r,t,tr,info=env.step(a)
        blocked=None
        for k in range(40):
            b,qprev=robot(o)
            a=np.zeros(11,dtype=np.float32); a[3+ji]=0.2*sign
            o,r,t,tr,info=env.step(a)
            b,q=robot(o)
            if np.allclose(q,qprev):
                blocked=qprev[ji]; break
        print("joint",ji+1,"sign",sign,"stopped at",None if blocked is None else round(float(blocked),3),
              "final q",np.round(q,2), "modelEE", np.round(kin.fk(q,base_x=b[0],base_y=b[1],base_rot=b[2])[:3,3],3))
env.close()
