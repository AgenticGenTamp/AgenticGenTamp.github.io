import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=1)
c=kutil.Ctl(env,o)
cu=c.opos("cube0")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang)
b,q,g=c.robot(); axb,ayb=c.armbase(b)
R=c.o.get_object_from_name("robot")
F=["grasp_tf_x","grasp_tf_y","grasp_tf_z","grasp_tf_qx","grasp_tf_qy","grasp_tf_qz","grasp_tf_qw","finger_state"]
def tf(): return np.array([float(c.o.get(R,f)) for f in F])
def fkp(q,mz): return kin.fk(q,base_x=axb,base_y=ayb,base_rot=b[2],mount=(0,0,mz))
print("cube",np.round(cu,3))
for z in [0.35,0.2]:
    c.move_to([cu[0],cu[1],z])
    _,q2,_=c.robot()
    print("z",z,"tf",np.round(tf(),3))
    print("  fk245",np.round(fkp(q2,0.245)[:3,3],3),"fk269",np.round(fkp(q2,0.269)[:3,3],3))
    print("  R",np.round(fkp(q2,0.269)[:3,:3],2).tolist())
env.close()
