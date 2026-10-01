from env_client import make_env
import numpy as np
for seed in range(5):
    e=make_env();s,i=e.reset(seed=seed)
    print('SEED',seed,'INFO',repr(i),'MAX',e.max_steps,flush=True)
    for name in s.get_object_names():
        o=s.get_object_from_name(name)
        fs=e.observation_space.type_features[o.type]
        print(name,{f:round(float(s.get(o,f)),5) for f in fs},flush=True)
    for j in range(3):
        a=np.zeros(11);a[-1]=1
        s,r,t,tr,info=e.step(a)
        print('STEP',j,r,t,tr,repr(info),flush=True)
    e.close()
