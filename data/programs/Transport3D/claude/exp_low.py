import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
def setup(seed=0):
    env=make_env(); o,info=env.reset(seed=seed)
    c=kutil.Ctl(env,o)
    cu=c.opos("cube1")
    ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
    c.gotobase(bp[0],bp[1],ang); c.move_to([cu[0],cu[1],0.3]); c.move_to([cu[0],cu[1],cu[2]]); c.grip(True)
    c.move_to([cu[0],cu[1],0.65])
    return c
c=setup()
print("grasp",c.robot()[2])
b,q,g=c.robot(); ax,ay=c.armbase(b)
cp=[ax+0.30*np.cos(b[2]), ay+0.30*np.sin(b[2]), 0.65]
c.move_to(cp)
c.gotobase(0.17,0.11,0.0)
print("base",np.round(c.robot()[0],3),"cube",np.round(c.opos("cube1"),3))
for (px,py) in [(0.733,0.114),(0.6,0.114),(0.5,0.114),(0.733,0.114)]:
    ok1=c.move_to([px,py,0.65])
    zs=[]
    for z in np.arange(0.60,0.415,-0.01):
        ok=c.move_to([px,py,float(z)])
        if not ok: zs.append(round(float(z),3)); break
    lastz=c.opos("cube1")[2]
    print("place",(px,py),"hi",ok1,"blocked_at",zs,"cubez",round(float(lastz),3))
    # try exact
    ok=c.move_to([px,py,0.4255])
    print("   exact 0.4255:",ok,"cube",np.round(c.opos("cube1"),3))
    if ok:
        c.grip(False); print("   released",c.robot()[2]); break
    c.move_to([px,py,0.65])
env.close=None
