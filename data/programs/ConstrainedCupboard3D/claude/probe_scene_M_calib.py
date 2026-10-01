import numpy as np
from env_client import make_env
def feats(obs,name):
    o=obs.get_object_from_name(name); return dict(zip(obs.type_features[o.type],[float(v) for v in obs.data[o]]))
def servo(env,obs,base=None,arm=None,grip=0.0,steps=200):
    for i in range(steps):
        rb=feats(obs,'robot'); a=np.zeros(11,dtype=np.float32); err=0
        if base is not None:
            for k,(f,idx) in enumerate([('pos_base_x',0),('pos_base_y',1),('pos_base_rot',2)]):
                e=base[k]-rb[f]; err=max(err,abs(e)); a[idx]=np.clip(2*e,-0.1,0.1)
        if arm is not None:
            for j in range(7):
                e=arm[j]-rb[f'pos_arm_joint{j+1}']; err=max(err,abs(e)); a[3+j]=np.clip(2*e,-0.1,0.1)
        a[10]=grip
        obs,r,*_=env.step(a)
        if err<0.008: break
    return obs
env=make_env(); obs,info=env.reset(seed=0,options={'object_count':3})
obs=servo(env,obs,base=[0.5,0.0,0.0])
q=[0,0.9,3.14,-1.5,0,1.6,1.57]
obs=servo(env,obs,arm=q)
rb=feats(obs,'robot'); print("base",round(rb['pos_base_x'],3),round(rb['pos_base_y'],3),"joints",[round(rb[f'pos_arm_joint{k}'],2) for k in range(1,8)])
# place markers (rods) at known world coords, axis-aligned along y
marks={'cuboid_0':(rb['pos_base_x']+0.30, rb['pos_base_y'], 0.015),
       'cuboid_1':(rb['pos_base_x']+0.60, rb['pos_base_y'], 0.015),
       'cuboid_2':(rb['pos_base_x']+0.45, rb['pos_base_y'], 0.50)}
for n,(x,y,z) in marks.items():
    o=obs.get_object_from_name(n)
    obs.set(o,'x',x); obs.set(o,'y',y); obs.set(o,'z',z)
    obs.set(o,'qw',1.0); obs.set(o,'qx',0.0); obs.set(o,'qy',0.0); obs.set(o,'qz',0.0)
print("marks",{k:tuple(round(v,3) for v in vv) for k,vv in marks.items()})
print(env.render_state(state=obs,label="M_calib"))
env.close()
