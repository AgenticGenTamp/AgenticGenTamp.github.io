from env_client import make_env
import numpy as np
E=make_env();s,i=E.reset(seed=1);r=s.get_object_from_name('robot');c=s.get_object_from_name('cube_0');c0=np.array([s.get(c,f) for f in ('x','y','z')]);q=np.array([0,2.4,np.pi,.3,0,-.873,np.pi/2]);base=np.array([c0[0],c0[1]+.85,-np.pi/2])
def move(b,grip,steps):
 global s
 for k in range(steps):
  v=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(j) for j in range(1,8)]]);a=np.zeros(11);a[:3]=np.clip(b-v[:3],-.07,.07);a[3:10]=np.clip(q-v[3:],-.1,.1);a[10]=grip
  s,re,t,tr,i=E.step(a)
  if t: print('SUCCESS',b,grip,re,flush=True);return True
 return False
move(base,0,150);move(base,1,10);q[1]=1.2;move(base,1,40)
print('PICK',[(f,s.get(c,f)) for f in ['x','y','z']],flush=True)
# Deliver carried cube through all fixture boundaries.
for b in [[1.7,.4,-np.pi/2],[1.7,-.3,-np.pi/2],[1.7,-.8,-np.pi/2],[1.3,-.8,-np.pi/2],[1.25,.5,-np.pi],[1.25,1.4,-np.pi],[1.7,1.5,np.pi/2],[1.7,1.8,np.pi/2],[1.9,1.,0],[1.9,.4,0],[1.8,-.4,0]]:
 if move(np.array(b),1,80):break
 print('VISIT',b,'ROBOT',[round(s.get(r,f),3) for f in ['pos_base_x','pos_base_y','pos_base_rot']],'CUBE',[round(s.get(c,f),3) for f in ['x','y','z']],flush=True)
E.close()
