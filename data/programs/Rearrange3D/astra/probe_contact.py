import numpy as np,json
from env_client import make_env

def drive(e,o,goal,n):
 for _ in range(n):
  a=np.clip((goal-o[93:104])*2,-.1,.1);a[10]=goal[10]
  o,r,t,tr,i=e.step(a.astype(np.float32))
 return o,r,t

e=make_env();records=[]
for grip in (0.,1.):
 for wrist in (.73,-.5,1.5):
  o,_=e.reset(seed=0);orig=o.copy();g=o[93:104].copy();g[4]=-1;g[6]=-1.6;g[8]=wrist;g[10]=grip
  o,_,_=drive(e,o,g,80)
  g[1]=o[17];g[2]=0;o,_,_=drive(e,o,g,20)
  for x in np.arange(-.2,.41,.1):
   g[0]=x;o,r,t=drive(e,o,g,8)
   d=o[16:19]-orig[16:19];dc=o[32:35]-orig[32:35]
   print('grip,wrist,x',grip,wrist,round(x,2),'box_d',np.round(d,3),'can_d',np.round(dc,3),'base',np.round(o[93:96],3),'q',np.round(o[96:103],2),flush=True)
   if np.linalg.norm(d)>.01:
    records.append({'grip':grip,'wrist':wrist,'x':float(x),'obs':o.tolist(),'delta':d.tolist()})
    break
with open('contact_records.json','w') as f:json.dump(records,f)
e.close()
