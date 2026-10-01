import numpy as np, sys
from env_client import make_env
import approach as A
from fk import fk

def sweep(axis, fixed, zw, lo, hi, seed=0):
    env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':4})
    ap=A.GeneratedApproach(None,None,{}); ap.reset(obs,info)
    b,q,g=A.robot_state(obs)
    bins0={n:A.obj_pos(obs,n).copy() for n in obs.get_object_names() if n.startswith('bin')}
    start=np.array([fixed,lo]) if axis==1 else np.array([lo,fixed])
    for tw,z,n in [(start,0.60,110),(start,zw,50)]:
        for t in range(n):
            a,gw=ap._servo(b,q,tw,z,0.0)
            obs,r,te,tr,i=env.step(a); b,q,g=A.robot_state(obs)
    gw,p=ap.gripper_world(b,q)
    print('  start gw',np.round(gw,3))
    qfix=q.copy(); b0=b.copy()
    events=[]
    steps=int(abs(hi-lo)/0.008)
    for k in range(steps+1):
        v = lo + (hi-lo)*k/steps
        bt=b0.copy(); bt[axis] = b0[axis] + (v-lo)
        act=np.zeros(11,dtype=np.float32)
        db=bt-b; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
        act[0:3]=np.clip(db/0.87,-0.1,0.1)
        dq=(qfix-q+np.pi)%(2*np.pi)-np.pi
        act[3:10]=np.clip(0.45*dq/0.249,-0.1,0.1)
        obs,r,te,tr,i=env.step(act); b,q,g=A.robot_state(obs)
        gw,_=ap.gripper_world(b,q)
        for n,p0 in bins0.items():
            d=np.linalg.norm(A.obj_pos(obs,n)[:2]-p0[:2])
            if d>0.006 and n not in [e[0] for e in events]:
                events.append((n, np.round(gw,3), round(float(d),3)))
    env.close(); return events
print('sweep +y at x=-0.1, z=0.46:', sweep(1, -0.1, 0.46, -0.32, 0.32))
print('sweep +x at y=0.15, z=0.46:', sweep(0, 0.15, 0.46, -0.32, 0.32))
