import numpy as np
from env_client import make_env
def feats(obs,name):
    o=obs.get_object_from_name(name); return dict(zip(obs.type_features[o.type],[float(v) for v in obs.data[o]]))
def servo(env,obs,base=None,arm=None,grip=0.0,steps=200):
    r=None
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
        if err<0.01: break
    return obs,r
env=make_env(); obs,info=env.reset(seed=0,options={'object_count':1})
c0=feats(obs,'cuboid_0')
q=[0,0.9,3.14,-1.5,0,1.6,1.57]
obs,r=servo(env,obs,base=[c0['x']-0.7,c0['y']-0.4,0.0])
obs,r=servo(env,obs,arm=q)
found=[]
for dx in [-0.7,-0.6,-0.5,-0.4,-0.3,-0.2]:
    for dy in np.arange(-0.4,0.41,0.1):
        obs,r=servo(env,obs,base=[c0['x']+dx,c0['y']+dy,0.0],arm=q,steps=40)
        c=feats(obs,'cuboid_0')
        d=abs(c['x']-c0['x'])+abs(c['y']-c0['y'])+abs(c['z']-c0['z'])
        if d>0.001: found.append((round(dx,2),round(dy,2),round(d,4),r))
print("contacts:",found if found else "NONE over 6x9 base grid with low-forward arm pose")
print("rod final",{k:round(v,4) for k,v in feats(obs,'cuboid_0').items() if k in 'xyz'},"r",r)
env.close()
