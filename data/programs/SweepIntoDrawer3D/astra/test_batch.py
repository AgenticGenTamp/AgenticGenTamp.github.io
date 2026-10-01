from env_client import make_env
from approach import GeneratedApproach
import numpy as np,sys,time
for seed in map(int,sys.argv[1:]):
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});start=time.monotonic();p.reset(s,i)
 for t in range(e.max_steps):
  s,r,d,tr,i=e.step(p.get_action(s))
  if d or tr:break
 print(seed,t+1,bool(d),'drawer',round(float(s[107]),3),'xyz',s[:80].reshape(5,16)[:,:3].round(3).tolist(),'time',round(time.monotonic()-start,1),flush=True)
 np.save('batch_%s.npy'%seed,s);e.close()
