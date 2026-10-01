import numpy as np,time,sys,json
from env_client import make_env
from approach import GeneratedApproach
start_seed=int(sys.argv[1]) if len(sys.argv)>1 else 10
count=int(sys.argv[2]) if len(sys.argv)>2 else 50
forced=int(sys.argv[3]) if len(sys.argv)>3 else None
e=make_env();results=[]
for seed in range(start_seed,start_seed+count):
 try:s,info=e.reset(seed=seed,options={'object_count':forced} if forced is not None else None)
 except Exception as ex:print('RESETERROR',seed,str(ex),flush=True);continue
 p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);started=time.time();same=0
 for i in range(160):
  a=p.get_action(s);ns,r,te,tr,inf=e.step(a)
  same=same+1 if np.max(np.abs(p.config(ns)-p.config(s)))<1e-6 else 0
  s=ns
  if te or tr or same>8:break
 result=[seed,info['object_count'],i+1,te,p.phase,round(time.time()-started,3)]
 results.append(result);print(result,flush=True)
 if not te:
  print('FAILDETAIL',p.target.name if p.target else None,p.xyz(s,p.target).tolist() if p.target else None,'drop',getattr(p,'drop',None),'config',p.config(s),flush=True)
print('TOTAL',sum(r[3] for r in results),'/',len(results),flush=True)
e.close()
