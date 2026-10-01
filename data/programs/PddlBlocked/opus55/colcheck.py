import sys, numpy as np
from env_client import make_env
from envutil import Sim
from planner import make_plan, TABLE
from collide import Obstacles, collisions
seed=int(sys.argv[1])
env=make_env(); S=Sim(env,env.reset(seed=seed)[0])
blk=S.block('blocker'); g0=S.block('green0')
def yaw(b): x,y,z,w=b[3:7]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
boxes=[(b[0],b[1],b[2],0.035,0.035,0.07,yaw(b)) for b in (blk,g0)]
obs=Obstacles(TABLE,boxes,[])
segs=make_plan(S.base(),S.q(),blk,g0,g0[2])
for w in segs:
    skip=set()
    if w['name'] in ('grasp_b','aside'): skip={0}
    if w['name'] in ('grasp_g','out','drop','carry','carry_lift'): skip={1}
    if w['name'] in ('aside','retract','axis','grasp_g','out','drop'): skip|={0}
    hits=[(k,collisions(b,q,obs,skip)) for k,(b,q) in enumerate(w['configs'])]
    hits=[h for h in hits if h[1]]
    print(w['name'],len(w['configs']),hits[:3])
