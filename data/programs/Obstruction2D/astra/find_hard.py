from env_client import make_env
import json
E=make_env();hard=[]
for seed in range(400,2400):
 s,info=E.reset(seed=seed);g=s.get_object_from_name('target_surface');b=s.get_object_from_name('target_block')
 sx,sw=s.get(g,'x'),s.get(g,'width');tw,th=s.get(b,'width'),s.get(b,'height')
 left=min(sx+(sw-tw)/2,max(sx+.001,1.49-tw/2));right=left+tw
 score=0
 obs=[]
 for o in s.get_objects(E.observation_space.get_type('rectangle')):
  if o==b or s.get(o,'static'):continue
  x,w,h=s.get(o,'x'),s.get(o,'width'),s.get(o,'height')
  if x+w>left-.1 and x<right+.1:
   score=max(score,x-1.5 if x>1.5 else 0, (h-th-.1) if (x>=right or x+w<=left) else 0)
  obs.append((x,w,h))
 if score>0:hard.append((score,seed,info,sx,sw,tw,th,obs))
 if seed%200==0:print('progress',seed,'hard',len(hard),flush=True)
hard.sort(reverse=True)
with open('hard_seeds.json','w') as f:json.dump(hard,f)
print('HARDEST',hard[:12],flush=True)
E.close()
