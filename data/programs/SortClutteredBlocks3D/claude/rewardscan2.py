import numpy as np, sys
from env_client import make_env
import approach as A
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
zw=float(sys.argv[2]) if len(sys.argv)>2 else 0.50
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':4})
ap=A.GeneratedApproach(None,None,{}); ap.reset(obs,info)
b,q,g=A.robot_state(obs)
# grasp cube1 using state machine until phase 'tobin'
t=0
while ap.phase!='tobin' and t<300:
    a=ap.get_action(obs); obs,r,te,tr,i=env.step(a); b,q,g=A.robot_state(obs); t+=1
print('grasped? cube1',np.round(A.obj_pos(obs,'cube1'),3), 'steps',t, flush=True)
# lift to zw
for k in range(40):
    a,gw=ap._servo(b,q,A.obj_pos(obs,"cube1")[:2],zw,ap.gclose,False)
    obs,r,te,tr,i=env.step(a); b,q,g=A.robot_state(obs)
qfix=q.copy()
res=[]
xs=np.arange(-0.34,0.35,0.04); ys=np.arange(-0.46,0.47,0.04)
b0=b.copy()
for ix,x in enumerate(xs):
    yy = ys if ix%2==0 else ys[::-1]
    for y in yy:
        # base target so that gripper world = (x,y)
        p=A.fk(qfix)[:3,3]
        bt=np.array([x + A.MOUNT[0]+p[0], y + A.MOUNT[1]+p[1], np.pi])
        for k in range(2):
            act=np.zeros(11,dtype=np.float32)
            db=bt-b; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
            act[0:3]=np.clip(db/0.87,-0.1,0.1)
            dq=(qfix-q+np.pi)%(2*np.pi)-np.pi
            act[3:10]=np.clip(0.45*dq/0.249,-0.1,0.1); act[10]=ap.gclose
            obs,r,te,tr,i=env.step(act); b,q,g=A.robot_state(obs)
        c=A.obj_pos(obs,'cube1')
        res.append((round(r,4), np.round(c,3)))
        if abs(r+1.0)>1e-6:
            print('REWARD CHANGE', r, 'cube at', np.round(c,3), flush=True)
print('distinct rewards', sorted(set(x[0] for x in res)))
print('cube z range', min(x[1][2] for x in res), max(x[1][2] for x in res))
env.close()
