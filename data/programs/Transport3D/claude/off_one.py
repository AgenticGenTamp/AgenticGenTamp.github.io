import numpy as np, kutil, kin, off_lib as L
from env_client import make_env
np.set_printoptions(precision=4,suppress=True)
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
cu=c.opos("cube1")
c.gotobase(-0.15,-0.24,3.13); b,q,g=c.robot(); yaw=b[2]
s0=c.steps
ok=L.safe_to(c,cu[0],cu[1],cu[2],yaw)
T,q,b=L.fkpos(c)
print("reach",ok,"fk",np.round(T[:3,3],4),"steps",c.steps-s0)
c.grip(True); print("grasp",c.robot()[2])
R=c.o.get_object_from_name("robot")
print("gtf",[round(float(c.o.get(R,"grasp_tf_"+f)),4) for f in ["x","y","z","qx","qy","qz","qw"]])
L.cmove(c,[T[0,3],T[1,3],0.20],yaw)
print("cube after lift",np.round(c.opos("cube1"),4),"total steps",c.steps)
env.close()
