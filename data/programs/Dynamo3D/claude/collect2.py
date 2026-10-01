import numpy as np, sys, json
from env_client import make_env
lo,hi=int(sys.argv[1]),int(sys.argv[2])
out=[]
for s in range(lo,hi):
    env=make_env(); obs,info=env.reset(seed=s)
    a=np.zeros(11,dtype=np.float32)
    for _ in range(30):
        obs,rew,term,trunc,info=env.step(a)
        if term: break
    names=sorted([n for n in obs.get_object_names() if n!='robot'])
    d={'seed':s,'n':info.get('object_count'),'term':bool(term),'rew':round(float(rew),4)}
    for n in names:
        o=obs.get_object_from_name(n)
        d[n]=[round(float(obs.get(o,f)),3) for f in ['x','y','z','vx','vy','vz']]
    out.append(d); env.close()
print(json.dumps(out))
