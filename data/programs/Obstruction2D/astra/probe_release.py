from env_client import make_env
import numpy as np

def act(e,s,a):return e.step(np.array(a))[0]
e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('robot');b=s.get_object_from_name('target_block');x=s.get(b,'x')+s.get(b,'width')/2
for i in range(6):s=act(e,s,[np.clip(x-s.get(r,'x'),-.05,.05),0,0,.1,0])
for i in range(55):s=act(e,s,[0,-.01,0,0,1])
for i in range(5):s=act(e,s,[0,.05,0,0,1])
print('LIFT',s.get(r,'y'),s.get(b,'y'))
for i in range(5):
 s=act(e,s,[0,0,0,0,0]);print('RELEASE',i,s.get(b,'y'))
for a in [[0,.05,0,0,0],[.05,0,0,0,0],[-.05,0,0,0,0]]:
 for i in range(40):s=act(e,s,a)
 print('LIMIT',a,s.get(r,'x'),s.get(r,'y'))
e.close()
# Find clear surface instance, transport while held to surface.
for seed in range(50):
 e=make_env();s,_=e.reset(seed=seed)
 if not any(n.startswith('obstruction') for n in s.get_object_names()):break
 e.close()
else:raise SystemExit('no empty scene')
print('CLEARSEED',seed);r=s.get_object_from_name('robot');b=s.get_object_from_name('target_block');g=s.get_object_from_name('target_surface')
def move(x,y,v=1):
 global s
 for _ in range(40):
  dx=np.clip(x-s.get(r,'x'),-.05,.05);dy=np.clip(y-s.get(r,'y'),-.05,.05)
  if abs(dx)+abs(dy)<1e-5:return False
  s,rr,t,tr,_=e.step(np.array([dx,dy,0,.1,v]))
  if t:print('SUCCESS',s.get(b,'x'),s.get(b,'y'),'vac',s.get(r,'vacuum'));return True
 return False
move(s.get(b,'x')+s.get(b,'width')/2,.8,0)
for _ in range(120):
 s=act(e,s,[0,-.005,0,.1,1])
move(s.get(r,'x'),.8,1)
move(s.get(g,'x')+s.get(g,'width')/2,.8,1)
move(s.get(r,'x'),.2,1)
print('END',s.get(b,'x'),s.get(b,'y'));e.close()
