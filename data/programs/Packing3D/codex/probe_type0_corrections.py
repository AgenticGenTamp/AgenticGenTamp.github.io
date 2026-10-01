"""Search grasp points whose held triangle can be seated without finger collision."""
from env_client import make_env
from probe_pick_place import JOINTS, OFFSET, g, move
import numpy as np

for dx,dy in [(x,y) for x in (0,-.015,.015,-.03,.03,-.045,.045)
                    for y in (0,-.015,.015,-.03,.03,-.045,.045)]:
    env=make_env(); s,_=env.reset(seed=1,options={'object_count':1})
    rob=s.get_object_from_name('robot'); p=s.get_object_from_name('part0'); rack=s.get_object_from_name('rack')
    xy=np.array([g(s,p,'pose_x'),g(s,p,'pose_y')]); corr=np.array([dx,dy])
    base=xy-OFFSET+corr
    s,*_=move(env,s,rob,base,grip=1.,steps=5); s,*_=move(env,s,rob,base,grip=-1.,steps=1)
    if not g(s,rob,'grasp_active'): env.close(); continue
    lifted=JOINTS.copy();lifted[1]-=.15
    s,*_=move(env,s,rob,base,joints=lifted,grip=0.,steps=1)
    rackxy=np.array([g(s,rack,'pose_x'),g(s,rack,'pose_y')])
    place=rackxy+[0,-.07]-OFFSET-[.0235403,0]+corr
    s,*_=move(env,s,rob,place,joints=lifted,grip=0.,steps=5)
    term=False
    for _ in range(35):
        a=np.zeros(11,np.float32);a[4]=.02
        s,r,term,trunc,_=env.step(a)
        if term or not g(s,rob,'grasp_active'):break
    if not term and g(s,rob,'grasp_active'):
        for _ in range(8):
            a=np.zeros(11,np.float32);a[6]=-.02;a[10]=1
            s,r,term,trunc,_=env.step(a)
            if term or not g(s,rob,'grasp_active'):break
    print('corr',dx,dy,'term',term,'held',g(s,rob,'grasp_active'),'pose',*[round(g(s,p,'pose_'+q),3) for q in 'xyz'],flush=True)
    env.close()
    if term:break
