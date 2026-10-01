import numpy as np
from env_client import make_env

def run(g):
 e=make_env(); o,_=e.reset(seed=0); p=o[147:150].copy()
 for k in range(20):
  a=np.zeros(11,np.float32); a[10]=g; o,r,*_=e.step(a)
 print('g',g,'grip',o[135],'wiper pre',o[147:150]-p)
 for k in range(15):
  a=np.zeros(11,np.float32); a[1]=-.1; a[10]=g; o,r,*_=e.step(a)
 print(' moved base',o[125:128],'wiper',o[147:150], 'delta',o[147:150]-p)
 e.close()
for g in [0.,1.]: run(g)
