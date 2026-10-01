import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
cu=c.opos("cube1")
c.gotobase(-0.15,-0.24,3.13)
b,q,g=c.robot(); yaw=b[2]
# path A: direct IK from retract-ish current q
qd=kin.ik_top_down(np.array([cu[0],cu[1],cu[2]]),yaw=yaw,q_init=q,base_x=c.armbase(b)[0],base_y=c.armbase(b)[1],base_rot=b[2],mount=(0,0,0.269))
print("direct qd",np.round(qd,3))
ok=c.goto(qd); c.grip(True); print("A reach",ok,"grasp",c.robot()[2])
b,q2,g2=c.robot(); print("q at A",np.round(q2,3))
if c.robot()[2]>0.5:
    c.grip(False)
# path B: cartesian chain like approach.py
qc=q2.copy()
cur=kin.fk(qc,base_x=c.armbase(b)[0],base_y=c.armbase(b)[1],base_rot=b[2],mount=(0,0,0.269))[:3,3]
print("cur ee",np.round(cur,3))
env.close()
