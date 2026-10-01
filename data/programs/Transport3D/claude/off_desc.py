import numpy as np, kutil, kin
from env_client import make_env
np.set_printoptions(precision=4,suppress=True)
kutil.MZ=0.269
def fgoto(c,qd,maxstep=0.05,nmax=400):
    for k in range(nmax):
        b,q,g=c.robot(); e=qd-q
        if np.abs(e).max()<1e-4: return True,k
        a=np.zeros(11); a[3:10]=np.clip(e,-maxstep,maxstep)
        qp=q.copy(); c.step(a); b,q,g=c.robot()
        if np.allclose(q,qp): return False,k
    return False,nmax
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
cu=c.opos("cube1")
c.gotobase(-0.15,-0.24,3.13)
b,q,g=c.robot(); yaw=b[2]
c.move_to([cu[0],cu[1],0.4],yaw=yaw)
for ms in [0.2,0.05,0.01]:
    qd=c.ik([cu[0],cu[1],cu[2]],yaw=yaw)
    ok,k=fgoto(c,qd,maxstep=ms)
    b,q,g=c.robot(); ax,ay=c.armbase(b)
    T=kin.fk(q,base_x=ax,base_y=ay,base_rot=b[2],mount=(0,0,0.269))
    print("maxstep",ms,"ok",ok,"k",k,"fk",np.round(T[:3,3],4),"err",np.round(np.abs(qd-q).max(),4))
c.grip(True); print("grasp",c.robot()[2])
env.close()
