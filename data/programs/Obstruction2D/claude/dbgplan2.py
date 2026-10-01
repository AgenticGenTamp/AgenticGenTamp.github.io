import numpy as np
from env_client import make_env
import approach as A
env=make_env(); ap=A.GeneratedApproach(env.action_space, env.observation_space, {})
obs,info=env.reset(seed=97, options={'object_count':4})
robot,rects,target,surface = ap.parse(obs)
lay={r.name:r.copy() for r in rects}
for n,x in [('obstruction0',1.205),('obstruction1',1.014),('obstruction2',1.28)]:
    lay[n].x=x; lay[n].y=A.PARK_BOTTOM
others=list(lay.values())
for tx in ap.target_x_candidates(target,surface):
    off=ap.find_move(target, others, tx, A.PLACE_BOTTOM)
    print("tx",round(tx,3),"off",off)
    if off is None:
        # diagnose
        for o in others: print("   ",o.name,round(o.x1,3),round(o.x2,3),round(o.top,3))
        gx=tx+target.w/2
        print("   y_place",ap.grasp_y(A.PLACE_BOTTOM+target.h),"min_y",ap.min_y(gx,gx,others,(target.h,target.w/2,target.w/2)))
        break
env.close()
