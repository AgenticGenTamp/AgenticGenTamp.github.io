from env_client import make_env
import numpy as np
from approach import fk, ik, down_R, ARM_OFFSET_X, GRASP_DZ
J=[f'joint_{i}' for i in range(1,8)]
def getq(o):
    r=o.get_object_from_name('robot'); return np.array([o.get(r,n) for n in J])
env=make_env()
for mode in ['close_with_last_move','close_then_lift_same']:
    obs,_=env.reset(seed=0); q0=getq(obs)
    c=obs.get_object_from_name('cube0'); r=obs.get_object_from_name('robot')
    pa=np.array([obs.get(c,'pose_x')-ARM_OFFSET_X, obs.get(c,'pose_y'), obs.get(c,'pose_z')-GRASP_DZ])
    R=down_R(np.pi/2)
    qg,e=ik(q0,pa,R); ql,e2=ik(qg,pa+[0,0,0.12],R)
    d=qg-q0; n=int(np.ceil(np.abs(d).max()/0.4)); print('steps needed',n, np.abs(d).max())
    for k in range(n):
        a=np.zeros(11,dtype=np.float32); a[3:10]=d/n
        if mode=='close_with_last_move' and k==n-1: a[10]=-1
        obs,rw,te,tr,_=env.step(a)
    print(mode,'q reached',np.abs(getq(obs)-qg).max()<1e-4,'grasp',obs.get(r,'grasp_active'))
    a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(ql-qg,-.4,.4); a[10]=-1
    obs,rw,te,tr,_=env.step(a)
    print(' after close+lift step: grasp',obs.get(r,'grasp_active'),'cz',obs.get(c,'pose_z'),'term',te, 'q moved', np.abs(getq(obs)-qg).max())
env.close()
