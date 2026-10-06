import numpy as np
from test_approach import make
from approach import *
env=make(); s,info=env.reset(seed=4, options={'object_count':30})
ap=GeneratedApproach(env.action_space, None, {}); ap.reset(s,info)
orig=ap._drop_cfgs
def w(chk,q0,rects,held,target_mode=False):
    out=orig(chk,q0,rects,held,target_mode)
    tgt=rects['target_block']; tc=rect_verts(tgt['x'],tgt['y'],tgt['th'],tgt['w'],tgt['h']).mean(0)
    for q in out[:3]:
        cs=[P.mean(0) for P in chk.held_polys(q)]
        print(' drop',held,np.round(q,2),'n',nsteps(q0,q),'held c',np.round(cs,2),'dt',np.round(np.linalg.norm(cs[0]-tc),2))
    return out
ap._drop_cfgs=w
for t in range(400):
    a=ap.get_action(s); s,r,term,trunc,_=env.step(a)
    if term: print("SUCCESS",t+1); break
