import numpy as np
from env_client import make_env
def feats(obs,name):
    o=obs.get_object_from_name(name); return dict(zip(obs.type_features[o.type],[float(v) for v in obs.data[o]]))
def servo(env,obs,base=None,arm=None,grip=0.0,steps=80):
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
        obs,r,term,trunc,info=env.step(a)
        if err<0.008: break
    return obs,r
P={"P1":[0,0.5,3.14,-2.0,0,1.2,1.57],
   "P2":[0,0.8,3.14,-1.6,0,1.5,1.57],
   "P3":[0,1.2,3.14,-1.2,0,1.8,1.57],
   "P4":[0,0.3,3.14,-2.4,0,1.9,1.57]}
env=make_env()
paths=[]
for name,q in P.items():
    obs,info=env.reset(seed=0,options={'object_count':1})
    c0=feats(obs,'cuboid_0')
    obs,r=servo(env,obs,base=[c0['x']-0.75,c0['y'],0.0],steps=200)
    obs,r=servo(env,obs,arm=q,grip=0.0,steps=150)
    rb=feats(obs,'robot')
    print(name,"joints reached",[round(rb[f'pos_arm_joint{k}'],2) for k in range(1,8)])
    paths.append(env.render_state(state=obs,label=f"K_{name}"))
    moved=None
    for k in range(22):
        bx=c0['x']-0.75+0.05*k
        obs,r=servo(env,obs,base=[bx,c0['y'],0.0],arm=q,grip=0.0,steps=12)
        c=feats(obs,'cuboid_0')
        d=abs(c['x']-c0['x'])+abs(c['y']-c0['y'])+abs(c['z']-c0['z'])
        if d>0.001:
            moved=(round(bx,2),round(d,4),round(c['x'],4),round(c['y'],4),round(c['z'],4),r)
            print("  MOVED at base_x",moved); break
    if moved is None: print("  no rod motion over base_x sweep; r=",r)
print("PATHS",paths)
env.close()
