from env_client import make_env
from grasp_baseline import GeneratedApproach
from calibration_math import ik
import numpy as np,sys,concurrent.futures
close=float(sys.argv[1]) if len(sys.argv)>1 else 1

def trial(arg):
 mount,dx=arg;e=make_env();s,i=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.mount=mount;p.close=close;p.reset(s,i);p.b[0]+=dx;mx=0;shift=0;low=None
 for k in range(310):
  s,r,t,tr,i=e.step(p.get_action(s));mx=max(mx,s[2]);shift=max(shift,np.linalg.norm(s[:2]-[.714351,-.233207]));
  if p.phase==2: low=s[:3].copy()
  if p.phase==3 and p.age>65:break
 print('RESULT',close,mount,dx,'z',round(mx,4),'shift',round(shift,4),'xyz',np.round(s[:3],4),'low',low,flush=True);e.close()
 return mx
args=[(m,x) for m in [.36,.40,.44] for x in [-.10,-.14,-.18]]
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:list(pool.map(trial,args))
