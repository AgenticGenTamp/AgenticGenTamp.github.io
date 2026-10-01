import numpy as np
from env_client import make_env
def feats(obs,name):
    o=obs.get_object_from_name(name); return dict(zip(obs.type_features[o.type],[float(v) for v in obs.data[o]]))
env=make_env(); obs,info=env.reset(seed=0,options={'object_count':1})
c0=feats(obs,'cuboid_0'); print("rod",round(c0['x'],3),round(c0['y'],3))
# align base behind rod
for i in range(200):
    rb=feats(obs,'robot'); a=np.zeros(11,dtype=np.float32)
    a[0]=np.clip(2*(c0['x']-0.75-rb['pos_base_x']),-0.1,0.1); a[1]=np.clip(2*(c0['y']-rb['pos_base_y']),-0.1,0.1)
    obs,r,*_=env.step(a)
q=[0,1.2,3.14,-1.2,0,1.8,1.57]
for i in range(200):
    rb=feats(obs,'robot'); a=np.zeros(11,dtype=np.float32); err=0
    for j in range(7):
        e=q[j]-rb[f'pos_arm_joint{j+1}']; err=max(err,abs(e)); a[3+j]=np.clip(2*e,-0.1,0.1)
    obs,r,*_=env.step(a)
    if err<0.008: break
xs=[]
for i in range(40):
    a=np.zeros(11,dtype=np.float32); a[0]=0.1
    obs,r,*_=env.step(a); rb=feats(obs,'robot'); c=feats(obs,'cuboid_0')
    xs.append((round(rb['pos_base_x'],3), round(c['x'],4)))
print("base_x traj:", xs[::4])
print("rod end", round(feats(obs,'cuboid_0')['x'],4), "r",r)
# now try pure joint2 sweep to slam arm down through the rod
for i in range(40):
    a=np.zeros(11,dtype=np.float32); a[3+1]=0.1
    obs,r,*_=env.step(a)
rb=feats(obs,'robot'); print("after j2 slam joints",[round(rb[f'pos_arm_joint{k}'],2) for k in range(1,8)],"rod",round(feats(obs,'cuboid_0')['x'],4),round(feats(obs,'cuboid_0')['z'],4),"r",r)
print(env.render_state(state=obs,label="L_slam"))
env.close()
