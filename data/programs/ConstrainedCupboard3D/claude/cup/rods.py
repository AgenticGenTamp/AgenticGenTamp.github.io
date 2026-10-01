import sys; sys.path.insert(0,"/sandbox")
import numpy as np, json
from env_client import make_env
env=make_env()
cfg=[('s0',0,None),('s1',1,None),('s7',7,None),('oc3_seed1',1,3),('oc3_seed2',2,3),('oc3_seed3',3,3),('oc3_seed11',11,3)]
out={}
for k,s,oc in cfg:
    o,_=env.reset(seed=s, options=({'object_count':oc} if oc else None))
    rods=[]
    for n in sorted(o.get_object_names()):
        if n.startswith('cuboid'):
            d=o.data[o.get_object_from_name(n)]
            rods.append([float(d[0]),float(d[1]),float(d[2]),float(d[3]),float(d[6])])
    rb=o.data[o.get_object_from_name('robot')]
    out[k]={'rods':rods,'base':[float(rb[0]),float(rb[1]),float(rb[2])]}
env.close()
json.dump(out,open('/sandbox/cup/rods.json','w'),indent=0)
print(json.dumps(out))
