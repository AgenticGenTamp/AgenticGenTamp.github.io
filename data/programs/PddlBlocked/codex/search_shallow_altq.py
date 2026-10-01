"""Search redundant arm IK at the feasible north/east tangent."""
import math,sys
import numpy as np
from scipy.optimize import least_squares
from env_client import make_env
from approach import GeneratedApproach
from probe_grasp import fk

LO=np.array([-.715,-.524,-.8,-2.321,-math.pi,-2.094,-math.pi])
HI=np.array([2.285,1.396,3.9,0,math.pi,0,math.pi])
def g(s,n,f):return float(s.get(s.get_object_from_name(n),f))
def start(seed):
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 for _ in range(80):
  if p.stage==5:break
  s,*_=e.step(p.get_action(s))
 # Let stage5 move base with lifted arm, but stop before its rejected lowering.
 for _ in range(20):
  old=p.robot(s);s,*_=e.step(p.get_action(s))
  if max(abs(p.robot(s)-old))<1e-6:break
 return e,s,p
def test(seed,q):
 e,s,p=start(seed)
 for _ in range(30):
  a=np.zeros(11,np.float32)
  for j in range(7):
   d=q[j]-g(s,'robot','joint_'+str(j+1));d=p.w(d) if j in (4,6) else d;a[3+j]=np.clip(d,-.08,.08)
  a[10]=-1;s,*_=e.step(a)
  if g(s,'green0','grasp_active')>.5:e.close();return True
 e.close();return False

seed=int(sys.argv[1]) if len(sys.argv)>1 else 101
e,s,p=start(seed);base=np.r_[p.robot(s),g(s,'robot','base_rot')];e.close()
T=fk(base,p.Q); pos=T[:3,3];axis=T[:3,0]; vertical=T[:3,2]
rng=np.random.default_rng(328)
seen=[]
def fun(q):
 X=fk(base,q);return np.r_[15*(X[:3,3]-pos),5*np.cross(X[:3,0],axis),2*np.dot(X[:3,2],axis)]
for k in range(250):
 z=least_squares(fun,rng.uniform(LO,HI),bounds=(LO,HI),max_nfev=300)
 q=z.x;err=np.linalg.norm(fun(q))
 if err>.03 or min([np.linalg.norm(q-x) for x in seen] or [9])<.15:continue
 seen.append(q);print('candidate',len(seen),'err',err,'q',q.tolist(),flush=True)
 if test(seed,q):print('HIT',seed,q.tolist(),flush=True);break
else:print('NONE',len(seen))
