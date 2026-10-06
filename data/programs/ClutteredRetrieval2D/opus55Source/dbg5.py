import numpy as np
from test_approach import make
from approach import *
env=make(); s,info=env.reset(seed=1, options={'object_count':30})
ap=GeneratedApproach(env.action_space, None, {}); ap.reset(s,info)
r,rects=ap._parse(s); m=ap._robot_model(r)
n='obstruction25'; d=rects[n]
chk0=Checker(m,ap._obstacles(rects,set()),[])
obs_n=ap._obstacles(rects,{n})
gs=ap._grasp_cfgs(d,m)
print(len(gs), 'free',sum(chk0.free(g) for g in gs), 'clean',sum(ap._suction_clean(m,obs_n,g) for g in gs), 'world', sum(chk0.colliders(g,0.01)[1] for g in gs))
sets=ap._blocker_sets(m,rects,gs,None,retreat_tgt=d,exclude={n,'target_block'})
print(len(sets), sorted(sets,key=len)[:6])
print(ap._rank(rects,sets,d))
for k in ['obstruction25','obstruction12','obstruction13','obstruction26','target_region']:
    dd=rects[k]; print(k, np.round(rect_verts(dd['x'],dd['y'],dd['th'],dd['w'],dd['h']).mean(0),3), round(dd['w'],3), round(dd['h'],3), round(dd['th'],2))
import time
q0=ap._q(r)
for k in ['obstruction24','obstruction10']:
    dd=rects[k]; print(k, np.round(rect_verts(dd['x'],dd['y'],dd['th'],dd['w'],dd['h']).mean(0),3))
    t=time.time(); p=ap._grasp_obstruction_path(m,rects,chk0,q0,k,False); print(' good', p is not None, time.time()-t)
    t=time.time(); p=ap._grasp_obstruction_path(m,rects,chk0,q0,k,True); print(' bad', p is not None, time.time()-t)
t=time.time(); p,nn=ap._try_remove(m,rects,chk0,q0,n,1,time.time()+6); print('try_remove', p is not None, nn, time.time()-t)
