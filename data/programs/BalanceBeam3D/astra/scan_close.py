from env_client import make_env
import numpy as np,json
records=[]
e=make_env()
for seed in range(200,400):
 s,_=e.reset(seed=seed)
 points=np.array([s[:2],s[54:56],s[70:72]])
 d=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)+np.eye(3)*10
 records.append((float(d.min()),seed,float(d[1,2])))
e.close()
records.sort()
json.dump(records,open('close_seed_ranking.json','w'))
print('closest_any',records[:8],flush=True)
print('closest_small',sorted(records,key=lambda r:r[2])[:8],flush=True)
