from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
names = obs.get_object_names()
R = obs.get_object_from_name('robot')
print("robot:", {k: round(float(obs.get(R,k)),4) for k in ('x','y','theta','base_radius','arm_joint','arm_length','gripper_base_width','gripper_base_height','finger_gap','finger_height','finger_width')})
xs=[];ys=[]
for n in names:
    if n=='robot': continue
    o=obs.get_object_from_name(n)
    try:
        x=float(obs.get(o,'x')); y=float(obs.get(o,'y'))
    except Exception as e:
        print(n,"ERR",e); continue
    xs.append(x); ys.append(y)
    if len(xs)<=6 or n=='hook':
        feats={}
        for f in ('x','y','theta','width','height','radius','side','length'):
            try: feats[f]=round(float(obs.get(o,f)),3)
            except Exception: pass
        print(n, feats)
print("n_obj",len(xs),"xrange",round(min(xs),3),round(max(xs),3),"yrange",round(min(ys),3),round(max(ys),3))
env.close()
