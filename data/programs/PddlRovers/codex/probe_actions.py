import numpy as np
from env_client import make_env

C = [-5/6, -1/2, -1/6, 1/6, 1/2, 5/6]
def get(st,n,f): return st.get(st.get_object_from_name(n),f)
def step(e,s,a): return e.step(np.asarray(a,dtype=np.float32))[0]
def act0(dx=0,dy=0,op=3): return [dx,dy,0,C[op], 0,0,0,C[3]]

for seed in range(6):
 e=make_env(); s,_=e.reset(seed=seed)
 for _ in range(17): s=step(e,s,act0(0,.2))
 print('seed',seed,'pos',get(s,'rover0','x'),get(s,'rover0','y'))
 for j in range(4):
  s=step(e,s,act0(0,0,1))
  print(' cal',j,get(s,'rover0','calibrated'),end='')
  s=step(e,s,act0(0,0,2))
  ims=[get(s,n,'have_image_rover0') for n in sorted(n for n in s.get_object_names() if n.startswith('objective'))]
  print(' img',ims)
 e.close()
