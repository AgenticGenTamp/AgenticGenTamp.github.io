import numpy as np
from env_client import make_env

e=make_env(); s,_=e.reset(seed=1)
names=sorted(n for n in s.get_object_names() if n.startswith('cuboid'))
def xyz(n):
 o=s.get_object_from_name(n); return np.array([s.get(o,f) for f in ('x','y','z')],float)
orig={n:xyz(n) for n in names}; robot=s.get_object_from_name('robot')
rng=np.random.default_rng(1)
for block in range(50):
 # Open, explore smooth randomly selected velocity for 8 simulation steps.
 a=np.zeros(11,np.float32); a[3:10]=rng.choice([-.1,0,.1],7); a[10]=1
 for j in range(8):
  s,r,t,tr,info=e.step(a)
 moves={n:np.linalg.norm(xyz(n)-orig[n]) for n in names}
 if max(moves.values())>.002:
  q=[s.get(robot,f'pos_arm_joint{i}') for i in range(1,8)]
  print('HIT',block,j,'q',np.round(q,3),'moves',moves,'xyz',[(n,np.round(xyz(n),3)) for n in names],flush=True)
  e.close(); raise SystemExit
 if block%5==0:
  q=[s.get(robot,f'pos_arm_joint{i}') for i in range(1,8)]
  print(block,np.round(q,2),flush=True)
print('NO HIT')
e.close()
