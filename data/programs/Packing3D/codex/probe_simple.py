from env_client import make_env
import numpy as np

def val(s,n,f): return s.get(s.get_object_from_name(n),f)
for seed in [0,1,2]:
    env=make_env(); s,info=env.reset(seed=seed,options={"object_count":1})
    p=s.get_object_from_name("part0"); r=s.get_object_from_name("rack")
    dx=s.get(p,"pose_x")-s.get(r,"pose_x"); dy=s.get(p,"pose_y")-s.get(r,"pose_y")
    print("seed",seed,"delta",dx,dy)
    for a in ([np.clip(dx,-.2,.2),np.clip(dy,-.2,.2),0,0,0,0,0,0,0,0,1],
              [dx-np.clip(dx,-.2,.2),dy-np.clip(dy,-.2,.2),0,0,0,0,0,0,0,0,1],
              [0,0,0,0,0,0,0,0,0,0,-1]):
        s,reward,term,trunc,info=env.step(np.array(a,dtype=np.float32))
        print(" base",val(s,"robot","pos_base_x"),val(s,"robot","pos_base_y"),"f",val(s,"robot","finger_state"),"g",val(s,"robot","grasp_active"),"pg",val(s,"part0","grasp_active"),"pp",val(s,"part0","pose_x"),val(s,"part0","pose_y"),val(s,"part0","pose_z"))
    env.close()
