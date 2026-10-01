import numpy as np, kutil
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
bx=c.opos("box0"); hx,hy,hz=0.1,0.15,0.1
ang=np.arctan2(bx[1],bx[0]); bp=bx[:2]-0.55*np.array([np.cos(ang),np.sin(ang)])
print("drive",c.gotobase(bp[0],bp[1],ang))
# edge nearest the robot: choose sign along x
b,q,g=c.robot()
gp=np.array([bx[0]-hx, bx[1], bx[2]+hz])
print("pre",c.move_to([gp[0],gp[1],0.5],yaw=ang))
print("desc",c.move_to(gp,yaw=ang),"| grip")
c.grip(True); print("grasp",c.robot()[2])
R=c.o.get_object_from_name("robot")
print("tf",[round(float(c.o.get(R,f)),3) for f in ["grasp_tf_x","grasp_tf_y","grasp_tf_z"]])
print("lift",c.move_to([gp[0],gp[1],0.75],yaw=ang),"box",np.round(c.opos("box0"),3))
b,q,g=c.robot(); ax,ay=c.armbase(b)
print("carry",c.move_to([ax+0.35*np.cos(b[2]),ay+0.35*np.sin(b[2]),0.75],yaw=ang))
print("drive2",c.gotobase(0.13,0.0,0.0))
print("box",np.round(c.opos("box0"),3))
# place box center at (0.62,0.0): grasp point offset = (-hx,0,+hz) rel center
tgtc=np.array([0.62,0.0,0.4+hz])
gp2=tgtc+np.array([-hx,0,hz])
print("pre2",c.move_to([gp2[0],gp2[1],0.8],yaw=ang))
for dz in [0.002,0.004,0.008]:
    ok=c.move_to([gp2[0],gp2[1],gp2[2]+dz],yaw=ang)
    print("lower",dz,ok,"box",np.round(c.opos("box0"),3))
    if ok:
        c.grip(False)
        print("release grasp",c.robot()[2],"box",np.round(c.opos("box0"),3),"done",c.done)
        if c.robot()[2]<0.5: break
print("steps",c.steps,"done",c.done)
env.close()
