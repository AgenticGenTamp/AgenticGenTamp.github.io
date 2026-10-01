from env_client import make_env
from calibration_math import ik
import numpy as np,sys
mount=float(sys.argv[1]) if len(sys.argv)>1 else .46
grip=float(sys.argv[2]) if len(sys.argv)>2 else 1
q=ik(.55,.018,mount=mount)
e=make_env();s,_=e.reset(seed=0); orig=s[:3].copy(); prev=orig.copy()
print('Q',q,flush=True)
b=np.array([s[0]-.55,s[1],0]);a=np.zeros(11);a[10]=1-grip
for k in range(140):
 a[:3]=np.clip(b-s[16:19],-.06,.06);a[3:10]=np.clip(q-s[19:26],-.1,.1);s,*_=e.step(a)
print('settled',s[16:27],flush=True)
for dy in [-.12,-.06,0,.06,.12]:
 for dx in np.linspace(-.2,.2,13):
  b[:2]=orig[:2]-[.55,0]+[dx,dy]
  for j in range(7):
   a[:3]=np.clip(b-s[16:19],-.025,.025);a[3:10]=np.clip(q-s[19:26],-.1,.1);a[10]=grip;s,*_=e.step(a)
  if np.linalg.norm(s[:3]-prev)>.001:print('CONTACT',dx,dy,s[:3], 'base',s[16:19],flush=True);prev=s[:3].copy()
print('final',s[:3],flush=True);e.close()
