import numpy as np, sys, itertools
from env_client import make_env
import approach as A

def sweep(cmd, zw, seed=0, axis=1, span=0.09):
    env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':4})
    ap=A.GeneratedApproach(None,None,{}); ap.reset(obs,info)
    b,q,g=A.robot_state(obs)
    c0=A.obj_pos(obs,'cube1').copy()
    start=c0[:2].copy(); start[axis]-=span
    for tw,z,n in [(start,0.60,110),(start,zw,60)]:
        for t in range(n):
            a,gw=ap._servo(b,q,tw,z,cmd)
            obs,r,te,tr,i=env.step(a); b,q,g=A.robot_state(obs)
    gw,_=ap.gripper_world(b,q)
    ok = abs(gw[2]-zw)<0.01 and np.linalg.norm(gw[:2]-start)<0.01
    qfix=q.copy(); b0=b.copy(); hit=None
    steps=int(2*span/0.006)
    for k in range(steps+1):
        bt=b0.copy(); bt[axis]=b0[axis]+2*span*k/steps
        act=np.zeros(11,dtype=np.float32)
        db=bt-b; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
        act[0:3]=np.clip(db/0.87,-0.1,0.1)
        dq=(qfix-q+np.pi)%(2*np.pi)-np.pi
        act[3:10]=np.clip(0.45*dq/0.249,-0.1,0.1); act[10]=cmd
        obs,r,te,tr,i=env.step(act); b,q,g=A.robot_state(obs)
        c=A.obj_pos(obs,'cube1')
        if hit is None and np.linalg.norm(c[:2]-c0[:2])>0.004:
            gw,_=ap.gripper_world(b,q)
            hit=(round(float(gw[axis]-c0[axis]),3), round(float(gw[2]),3))
    env.close(); return ('startok' if ok else 'BADSTART'), hit
import sys
cmd=float(sys.argv[1]); zw=float(sys.argv[2])
print('cmd',cmd,'zw',zw, sweep(cmd,zw), flush=True)
