import numpy as np,json
from env_client import make_env
from kinematics_candidate import planar_ik,forward

def go(e,o,q,base,g,n):
 for _ in range(n):
  a=np.zeros(11,np.float32);a[:3]=np.clip((base-o[93:96])*2,-.1,.1);a[3:10]=np.clip((q-o[96:103])*2,-.1,.1);a[10]=g
  o,*_=e.step(a)
 return o
for g in [1]:
 for z in [.10,.14,.18]:
  e=make_env();o,_=e.reset(seed=0);base=np.array([o[16]-.77,o[17]-.001,0.]);p0=o[16:19].copy()
  qhigh=planar_ik(.65,.27,mount_height=0.)
  back=base.copy();back[0]=-.8
  o=go(e,o,o[96:103].copy(),back,1-g,35)
  o=go(e,o,qhigh,back,1-g,100)
  o=go(e,o,qhigh,base,1-g,35)
  print("prelower",g,z,o[16:19].tolist(),flush=True)
  q=planar_ik(.65,z,mount_height=0.)
  o=go(e,o,q,base,1-g,35)
  p1=o[16:19].copy();o=go(e,o,q,base,g,25)
  p2=o[16:19].copy();o=go(e,o,qhigh,base,g,50)
  row={'g':g,'z':z,'before':p0.tolist(),'lower':p1.tolist(),'close':p2.tolist(),'lift':o[16:19].tolist(),'finger':float(o[103]),'qerr':float(np.linalg.norm(qhigh-o[96:103]))}
  print(json.dumps(row),flush=True);e.close()
