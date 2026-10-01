from env_client import make_env
from approach import GeneratedApproach
import numpy as np,time,sys
E=make_env();s,i=E.reset(seed=int(sys.argv[1]) if len(sys.argv)>1 else 0,options={'object_count':int(sys.argv[2])} if len(sys.argv)>2 else None);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,i);old=-1;t0=time.monotonic()
for k in range(600):
 s,r,t,tr,i=E.step(p.get_action(s))
 if p.stage!=old or k%40==0:
  old=p.stage;cp=np.array([p.xyz(s,o) for o in p.cubes]);gp=p.xyz(s,p.green)
  print(k,p.stage,r,'src',p.xyz(s,p.source).round(3),'green',gp.round(3),'mean',cp.mean(0).round(3),'near',sum((abs(cp[:,0]-gp[0])<.22)&(abs(cp[:,1]-gp[1])<.15)&(cp[:,2]<gp[2]+.1)),flush=True)
 if t or tr:break
cp=np.array([p.xyz(s,o) for o in p.cubes]);print('FINAL',cp.min(0),cp.max(0),cp.mean(0),'green',p.xyz(s,p.green),'src',p.xyz(s,p.source),'info',i,flush=True)
print('END',k,r,t,tr,time.monotonic()-t0,flush=True);E.close()
