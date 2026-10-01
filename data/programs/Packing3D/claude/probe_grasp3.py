import numpy as np, sys
from probe_lib import *
env=make_env(); obs,info=env.reset(seed=0)
base=rb(obs); p=ppos(obs,'part0')
print("part0",p)
obs=grip(env,obs,1.0)
def trial(pos,R,label):
    global obs
    obs,_,_=goto(env,obs,(p[0],p[1],0.40),Rdown,base,maxsteps=40)
    obs,blk,msg=goto(env,obs,pos,R,base,maxsteps=60)
    f=fkpos(obs)
    obs=grip(env,obs,-1.0)
    ga=rfeat(obs,'grasp_active')
    print("%-28s %s fk=%s grasp=%.1f finger=%.2f"%(label,msg,np.round(f,3),ga,rfeat(obs,'finger_state')),flush=True)
    if ga>0.5:
        print("GRASP!! grasp_tf",[round(rfeat(obs,'grasp_tf_'+k),4) for k in ['x','y','z','qx','qy','qz','qw']])
        return True
    obs=grip(env,obs,1.0)
    return False
# yaw sweep top-down
for yawdeg in [0,15,30,45,60,90]:
    R=Rdown@rotz(np.radians(yawdeg))
    for z in [0.26,0.245]:
        if trial((p[0],p[1],z),R,"yaw%d z%.3f"%(yawdeg,z)): sys.exit(0)
# tilt sweep: rotate approach axis away from vertical, position offset along approach
for tiltdeg in [30,60,90]:
    a=np.radians(tiltdeg)
    for sgn,axname in [(1,'+x'),(-1,'-x')]:
        R=Rdown@roty(sgn*a)
        zt=R[:,2]
        for d in [0.0,0.05,0.10,0.15]:
            tp=np.array([p[0],p[1],0.095])-zt*d
            if tp[2]<0.10: continue
            if trial(tp,R,"tilt%d%s d%.2f"%(tiltdeg,axname,d)): sys.exit(0)
for tiltdeg in [90]:
    a=np.radians(tiltdeg)
    for sgn,axname in [(1,'+y'),(-1,'-y')]:
        R=Rdown@rotx(sgn*a)
        zt=R[:,2]
        for d in [0.0,0.05,0.10,0.15]:
            tp=np.array([p[0],p[1],0.095])-zt*d
            if tp[2]<0.10: continue
            if trial(tp,R,"tilt%d%s d%.2f"%(tiltdeg,axname,d)): sys.exit(0)
print("done no grasp")
env.close()
