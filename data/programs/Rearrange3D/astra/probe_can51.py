import numpy as np,json
from concurrent.futures import ThreadPoolExecutor
from env_client import make_env
from kinematics_candidate import planar_ik

def go(e,o,q,base,g,n):
 for _ in range(n):
  a=np.zeros(11,np.float32);a[:3]=np.clip((base-o[93:96])*2,-.1,.1);a[3:10]=np.clip((q-o[96:103])*2,-.1,.1);a[10]=g
  o,*_=e.step(a)
 return o

def trial(args):
 dx,z=args;e=make_env();o,_=e.reset(seed=51);base=np.array([o[32]-.65-dx,o[33]-.001,0.]);p0=o[32:35].copy()
 qhigh=planar_ik(.65,.27,mount_height=0.);qhigh[6]=2.925;back=base.copy();back[0]=-.8
 o=go(e,o,o[96:103].copy(),back,0,35)
 o=go(e,o,qhigh,back,0,100)
 o=go(e,o,qhigh,base,0,35)
 q=planar_ik(.65,z,mount_height=0.);q[6]=2.925
 o=go(e,o,q,base,0,35);p1=o[32:35].copy()
 o=go(e,o,q,base,1,25);p2=o[32:35].copy()
 o=go(e,o,qhigh,base,1,50)
 row={'dx':dx,'z':z,'before':p0.tolist(),'lower':p1.tolist(),'close':p2.tolist(),'lift':o[32:35].tolist(),'quaternion':o[35:39].tolist(),'qerr':float(np.linalg.norm(qhigh-o[96:103]))}
 print(json.dumps(row),flush=True);e.close();return row
with ThreadPoolExecutor(max_workers=2) as pool:
 rows=list(pool.map(trial,[(.12,.10)]))
with open('can51_results.json','w') as f:json.dump(rows,f)
