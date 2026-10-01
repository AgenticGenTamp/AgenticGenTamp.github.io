import numpy as np, kutil, kin
from env_client import make_env
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
tgt=c.opos("box0"); print("box",np.round(tgt,3))
ang=np.arctan2(tgt[1],tgt[0]); bp=tgt[:2]-0.5*np.array([np.cos(ang),np.sin(ang)])
print("drive",c.gotobase(bp[0],bp[1],ang))
for z in [0.3,0.25,0.2,0.175,0.15,0.125,0.1,0.075,0.05]:
    ok=c.move_to([tgt[0],tgt[1],z])
    c.grip(True)
    b,q,g=c.robot()
    print("z",z,"reach",ok,"grasp",g)
    if g>0.5:
        R=c.o.get_object_from_name("robot")
        print("tf",[round(float(c.o.get(R,f)),3) for f in ["grasp_tf_x","grasp_tf_y","grasp_tf_z"]])
        print("lift",c.move_to([tgt[0],tgt[1],0.6]),"box now",np.round(c.opos("box0"),3))
        break
    c.grip(False)
env.close()
