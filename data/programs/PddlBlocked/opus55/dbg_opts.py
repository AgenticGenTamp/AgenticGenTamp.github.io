import sys, numpy as np, time
from env_client import make_env
import approach as A, planner, collide
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=A.GeneratedApproach(env.action_space, env.observation_space, None); ap.reset(obs,info)
ap._names=[n for n in obs.get_object_names() if n=='blocker' or n.startswith('green')]
blk=ap._block('blocker'); g0=ap._block('green0')
d,perp,z=A.geom(blk,g0,g0[2]); print('d',d.round(2),'blk',blk[:2].round(3))
ap._pen=collide.pen_boxes(g0[0],g0[1],ap._yaw(g0)); ap._set_obs(skip={'blocker'})
import stages; stages.DEBUG=True
t=time.time()
for name,segs in A.blocker_options(ap._base(),blk,g0,g0[2]): print('OPT',name)
print('time',time.time()-t)
