import numpy as np, sys, json
from env_client import make_env
lo,hi=int(sys.argv[1]),int(sys.argv[2])
out=[]
for s in range(lo,hi):
    try:
        env=make_env(); obs,info=env.reset(seed=s)
        names=sorted([n for n in obs.get_object_names() if n!='robot'])
        d={'seed':s,'n':info.get('object_count')}
        for n in names:
            o=obs.get_object_from_name(n)
            d[n]=[round(float(obs.get(o,f)),3) for f in ['x','y','z','qw','qx','qy','qz']]
        r=obs.get_object_from_name('robot')
        d['robot']=[round(float(obs.get(r,f)),3) for f in ['pos_base_x','pos_base_y','pos_base_rot']]
        out.append(d); env.close()
    except Exception as e:
        print("err",s,e)
print(json.dumps(out))
