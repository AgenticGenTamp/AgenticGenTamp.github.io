from env_client import make_env
import numpy as np,json

e=make_env();records=[]
for seed in range(40,100):
 s,_=e.reset(seed=seed)
 records.append({'seed':seed,'positions':s[[0,1,2,16,17,18,32,33,34]].tolist(),'box_distance':float(np.linalg.norm(s[16:19]-s[:3])),'can_distance':float(np.linalg.norm(s[32:35]-s[:3])),'box_tilt':float(abs(s[19])),'can_tilt':float(abs(s[35]))})
e.close()
json.dump(records,open('initial_samples.json','w'))
for key in ['box_distance','can_distance','box_tilt','can_tilt']:
 for mode in [min,max]:
  r=mode(records,key=lambda r:r[key]);print(key,mode.__name__,r['seed'],r[key])
for j in [0,1,3,4,6,7]:
 for mode in [min,max]:
  r=mode(records,key=lambda r:r['positions'][j]);print('coord',j,mode.__name__,r['seed'],r['positions'][j])
