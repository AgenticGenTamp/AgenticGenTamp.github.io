import numpy as np
from env_client import make_env

def get(s,n,fs):
 o=s.get_object_from_name(n); return np.array([s.get(o,f) for f in fs],float)

e=make_env(); s,_=e.reset(seed=1); names=sorted(n for n in s.get_object_names() if n.startswith('cuboid'))
orig={n:get(s,n,('x','y','z')) for n in names}
def step(a,i):
 global s
 s,r,t,tr,inf=e.step(np.array(a,np.float32))
 moves={n:float(np.linalg.norm(get(s,n,('x','y','z'))-orig[n])) for n in names}
 rob=np.round(get(s,'robot',('pos_base_x','pos_base_y','pos_base_rot')),2)
 print(i,r,'base',rob,'moves',moves) if max(moves.values())>.005 or i%10==0 else None
 return max(moves.values())
# base to cuboid1 vicinity x=.35 y=.12
for i in range(5):
 a=np.zeros(11); a[0]=.1; a[1]=.05; step(a,i)
rng=np.random.default_rng(44)
for k in range(80):
 a=np.zeros(11); a[3:10]=rng.choice([-.1,.1],7); a[10]=(k//8)%2
 if step(a,k+5)>.05: break
e.close()
