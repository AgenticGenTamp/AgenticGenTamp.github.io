import numpy as np
from env_client import make_env
def feats(obs,name):
    o=obs.get_object_from_name(name); return dict(zip(obs.type_features[o.type],[float(v) for v in obs.data[o]]))
env=make_env()
for label,(dx,dy) in {'+x':(0.1,0),'-x':(-0.1,0),'+y':(0,0.1),'-y':(0,-0.1)}.items():
    obs,info=env.reset(seed=0,options={'object_count':1})
    last=None
    for i in range(120):
        a=np.zeros(11,dtype=np.float32); a[0]=dx; a[1]=dy
        obs,r,*_=env.step(a); rb=feats(obs,'robot')
        cur=(round(rb['pos_base_x'],3),round(rb['pos_base_y'],3))
        if cur==last: break
        last=cur
    print(label,"stopped at",last,"after",i,"steps r=",r, "rod",round(feats(obs,'cuboid_0')['x'],3),round(feats(obs,'cuboid_0')['y'],3))
env.close()
