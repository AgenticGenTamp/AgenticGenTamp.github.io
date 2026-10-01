import numpy as np,json
from env_client import make_env

def move(e,o,g,n):
 for _ in range(n):
  a=np.clip((g-o[93:104])*2,-.1,.1);a[10]=g[10];o,r,t,tr,i=e.step(a.astype(np.float32))
 return o
out=[]
for grip in (0.,1.):
 e=make_env();o,_=e.reset(seed=0);orig=o.copy();g=o[93:104].copy();g[0]=o[16]-.133;g[1]=o[17]-.001;g[2]=0;g[10]=grip
 o=move(e,o,g,60);before=o.copy();g[10]=1-grip;o=move(e,o,g,10);g[4]=-1.;o=move(e,o,g,30)
 print('grip',grip,'base',np.round(o[93:96],3),'box_delta',np.round(o[16:19]-orig[16:19],4),'arm',np.round(o[96:103],3),flush=True)
 out.append(o.tolist());e.close()
with open('home_grip_results.json','w') as f:json.dump(out,f)
