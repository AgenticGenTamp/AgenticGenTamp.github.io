from env_client import make_env
import numpy as np
mins=[];sp=[]
e=make_env()
for seed in range(150):
 s,_=e.reset(seed=seed,options={'object_count':8})
 t=e.observation_space.get_type('rectangle');walls={}
 for o in s.get_objects(t):
  if not o.name.startswith('obstacle'):continue
  x,y,w,h=[float(s.get(o,f)) for f in ['x','y','width','height']]
  walls.setdefault(x,[]).append((y,y+h,w))
 for x,rects in walls.items():
  rects.sort();mins.append((rects[1][0]-rects[0][1],seed,x))
 xs=sorted(walls);sp.extend(xs[i+1]-xs[i]-.01 for i in range(len(xs)-1))
print('Gap minimum',min(mins),'max',max(x[0] for x in mins),'percentiles',np.percentile([x[0] for x in mins],[0,5,50,95,100]))
print('corridor width',min(sp),max(sp))
e.close()
