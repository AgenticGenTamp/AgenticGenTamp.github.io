from env_client import make_env
from approach import GeneratedApproach
import sys,time,json
lo=int(sys.argv[1]) if len(sys.argv)>1 else 0
hi=int(sys.argv[2]) if len(sys.argv)>2 else 100
count=int(sys.argv[3]) if len(sys.argv)>3 else None
rows=[]
e=make_env()
for seed in range(lo,hi):
 s,info=e.reset(seed=seed,options={'object_count':count} if count else None);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);start=time.time()
 for k in range(e.max_steps):
  s,r,t,tr,i=e.step(p.get_action(s))
  if t or tr:break
 row=[seed,info.get('object_count'),k+1,t,time.time()-start];rows.append(row)
 if not t: print('FAIL',row,flush=True)
 if (seed-lo)%10==9:print('PROGRESS',seed+1,'solved',sum(z[3] for z in rows),'avgsteps',sum(z[2] for z in rows)/len(rows),flush=True)
e.close()
print('SUMMARY',len(rows),sum(z[3] for z in rows),'maxsteps',max(z[2] for z in rows),'maxsecs',max(z[4] for z in rows),flush=True)
with open('validation_'+str(lo)+'_'+str(hi)+'_'+str(count)+'.json','w') as f:json.dump(rows,f)
