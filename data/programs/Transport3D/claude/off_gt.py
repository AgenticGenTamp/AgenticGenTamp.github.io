import numpy as np, kutil, kin
from env_client import make_env
np.set_printoptions(precision=4,suppress=True)
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
def gt():
    R=c.o.get_object_from_name("robot")
    p=np.array([float(c.o.get(R,"grasp_tf_"+f)) for f in "xyz"])
    qt=np.array([float(c.o.get(R,"grasp_tf_q"+f)) for f in ["x","y","z","w"]])
    return p,qt
b,q,g=c.robot()
print("base",np.round(b,4)); print("q",np.round(q,4))
p,qt=gt(); print("gt pos",np.round(p,4),"quat",np.round(qt,4))
ax,ay=c.armbase(b)
for mz in [0.0,0.245,0.269]:
    T=kin.fk(q,base_x=ax,base_y=ay,base_rot=b[2],mount=(0,0,mz))
    print("fk mz=%.3f"%mz, np.round(T[:3,3],4))
T=kin.fk(q,base_x=0,base_y=0,base_rot=0,mount=(0,0,0))
print("fk local",np.round(T[:3,3],4))
print("fk R",np.round(T[:3,:3],3).tolist())
env.close()
