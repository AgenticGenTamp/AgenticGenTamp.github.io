import numpy as np
from concurrent.futures import ThreadPoolExecutor
from env_client import make_env

def run(case):
 x,h=case;env=make_env();s,_=env.reset(seed=0)
 for k in range(8):
  a=np.zeros(11,dtype=np.float32);a[:2]=np.clip([x-s[0],-1.8-s[1]],-.4,.4);a[2]=np.clip(h-s[2],-.4,.4)
  s,r,t,tr,i=env.step(a)
  if np.linalg.norm(s[:3]-[x,-1.8,h])<1e-5:break
 safe=-1.8;blocked=-2.15
 for k in range(20):
  y=(safe+blocked)/2
  a=np.zeros(11,dtype=np.float32);a[1]=y-s[1]
  n,r,t,tr,i=env.step(a)
  if abs(n[1]-y)<2e-7:safe=float(n[1])
  else:blocked=y
  s=n
 print('BOUNDARY',x,h,safe,blocked,flush=True);env.close()
 return x,h,safe
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,[(x,h) for x in (1.3,1.32) for h in (-.2,-.4,-.6,-.8)]))
for x in (1.3,1.32):
 rows=[a for a in results if a[0]==x];hs=np.array([a[1] for a in rows]);ys=np.array([a[2] for a in rows]);A=np.column_stack((-np.sin(hs),np.cos(hs),np.ones(len(hs))));b=ys*np.cos(hs)-x*np.sin(hs);coef=np.linalg.lstsq(A,b,rcond=None)[0]
 print('FIT',x,coef.tolist(),'residual',(A@coef-b).tolist(),flush=True)
