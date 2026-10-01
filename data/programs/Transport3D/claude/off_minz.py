import numpy as np, kutil, kin
from env_client import make_env
np.set_printoptions(precision=4,suppress=True)
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
cu=c.opos("cube1")
names=[]
try:
    for n in ["cube1","cube2","cube3","cube4","table","floor","target","goal"]:
        try:
            p=c.opos(n); names.append((n,np.round(p,3)))
        except Exception: pass
except Exception: pass
print("objs",names)
c.gotobase(-0.15,-0.24,3.13)
b,q,g=c.robot(); yaw=b[2]
def minz(x,y,tag):
    c.move_to([x,y,0.45],yaw=yaw)
    last=None
    for z in np.arange(0.40,-0.001,-0.02):
        ok=c.move_to([x,y,z],yaw=yaw)
        bb,qq,gg=c.robot(); ax,ay=c.armbase(bb)
        T=kin.fk(qq,base_x=ax,base_y=ay,base_rot=bb[2],mount=(0,0,0.269))
        if not ok:
            print(tag,"blocked cmd z=%.3f actual fk"%z,np.round(T[:3,3],4)); return
        last=T[:3,3]
    print(tag,"reached z=0 fk",np.round(last,4))
minz(cu[0],cu[1],"over-cube")
minz(cu[0],cu[1]-0.20,"empty1")
minz(cu[0]+0.10,cu[1]-0.20,"empty2")
env.close()
