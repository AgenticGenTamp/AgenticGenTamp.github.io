import numpy as np, kutil, kin, off_lib as L
from env_client import make_env
np.set_printoptions(precision=4,suppress=True)
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
cu=c.opos("cube1")
c.gotobase(-0.15,-0.24,3.13); b,q,g=c.robot(); yaw=b[2]
L.cmove(c,[cu[0],cu[1],0.28],yaw)
for z in [0.20,0.15,0.10,0.07,0.05,0.04,0.03,0.025]:
    b,q,g=c.robot(); ax,ay=c.armbase(b)
    qd_raw,i=kin.ik_top_down(np.array([cu[0],cu[1],z]),yaw=yaw,q_init=q,base_x=ax,base_y=ay,
                             base_rot=b[2],mount=(0,0,0.269),return_info=True)
    qd=c.unwrap(qd_raw,q)
    ok=c.goto(qd)
    T,q2,b2=L.fkpos(c)
    print("z=%.3f ikOK=%s err=%.5f goto=%s fk=%s dq=%.4f"%(z,i["success"],i["pos_err"],ok,
          np.round(T[:3,3],4),np.abs(qd-q2).max()))
env.close()
