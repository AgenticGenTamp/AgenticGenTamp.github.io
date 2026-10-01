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
cu=c.opos("cube1"); print("cube",np.round(cu,4))
c.gotobase(-0.15,-0.24,3.13)
b,q,g=c.robot(); yaw=b[2]
print("ok wp",c.move_to([cu[0],cu[1],0.4],yaw=yaw))
print("ok dn",c.move_to([cu[0],cu[1],cu[2]],yaw=yaw))
c.grip(True)
b,q,g=c.robot(); print("grasp_active",g)
p,qt=gt(); print("grasp_tf p",np.round(p,4),"q",np.round(qt,4))
cu2=c.opos("cube1"); print("cube now",np.round(cu2,4))
ax,ay=c.armbase(b)
T=kin.fk(q,base_x=ax,base_y=ay,base_rot=b[2],mount=(0,0,0.269))
print("fk pos",np.round(T[:3,3],4)); print("fk R",np.round(T[:3,:3],3).tolist())
print("q",np.round(q,4))
env.close()
