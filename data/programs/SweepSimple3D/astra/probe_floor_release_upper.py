from env_client import make_env
import numpy as np

e=make_env();s,i=e.reset(seed=1);co=s.get_object_from_name('cube_0');ro=s.get_object_from_name('robot');fs=['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(j) for j in range(1,8)]
def cv():return np.array([s.get(co,f) for f in ['x','y','z']])
orig=cv();base=np.array([orig[0],orig[1]-.85,np.pi/2]);q=np.array([0,2.4,np.pi,.3,0,-.873,np.pi/2]);k=0
for k in range(220):
 actual=np.array([s.get(ro,f) for f in fs]);a=np.zeros(11,dtype=np.float32);a[:3]=np.clip(base-actual[:3],-.07,.07);a[3:10]=np.clip((q if k>=60 else np.array([0,-.349,np.pi,-2.548,0,-.873,np.pi/2]))-actual[3:],-.1,.1);a[10]=int(k>=210);s,r,t,tr,i=e.step(a)
print('GRASP',cv().tolist(),'BASE',[s.get(ro,f) for f in fs[:3]],flush=True)
path=[]
for j,y in enumerate(np.arange(.7,1.601,.1)):
 xs=[1.35,1.45,1.55,1.65,1.75,1.85]
 if j%2:xs.reverse()
 path.extend((x,float(y)) for x in xs)
for idx,goal in enumerate(path):
 stuck=0
 for local in range(70):
  actual=np.array([s.get(ro,f) for f in fs]);delta=np.array(goal)-cv()[:2];a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(delta,-.02,.02);a[2]=np.clip(np.pi/2-actual[2],-.07,.07);a[3:10]=np.clip(q-actual[3:],-.1,.1);a[10]=1
  prev=cv();s,r,t,tr,i=e.step(a);k+=1
  if np.linalg.norm(cv()-prev)<.0001:stuck+=1
  else:stuck=0
  if t or tr or max(abs(delta))<.013 or stuck>=12:break
 if t or tr:break
 for grip in [0,0,0,0,1,1]:
  actual=np.array([s.get(ro,f) for f in fs]);a=np.zeros(11,dtype=np.float32);a[3:10]=np.clip(q-actual[3:],-.1,.1);a[:2]=np.clip(cv()[:2]-[0,.85]-actual[:2],-.02,.02);a[10]=grip;s,r,t,tr,i=e.step(a);k+=1
  if not grip:print('RELEASE',idx,goal,'STEP',k,'CUBE',np.round(cv(),4).tolist(),'R',r,'T',t,'TR',tr,flush=True)
  if t or tr:break
 if t or tr or k>=998:break
e.close()
