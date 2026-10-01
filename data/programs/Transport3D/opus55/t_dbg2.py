import sys, numpy as np, approach
from env_client import make_env
env=make_env(); ap=approach.GeneratedApproach(env.action_space, env.observation_space, {})
obs,info=env.reset(seed=int(sys.argv[1])); ap.reset(obs,info)
N=int(sys.argv[2]); cube=sys.argv[3]
for t in range(N):
    a=ap.get_action(obs)
    obs,r,term,trunc,info=env.step(a)
    base,q=ap._robot(); p,R=ap._ee_world(base,q); o=ap._obj(cube)
    if a[10]!=0 or t%4==0: print(t,'grip',a[10],'ga',ap._grasped(),'base',np.round(base,3),'ee',np.round(p,3),'cube',round(o['x'],3),round(o['y'],3),round(o['z'],3), 'yaw',round(approach.quat_yaw(*o['q']),3))
