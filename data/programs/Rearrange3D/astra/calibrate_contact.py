import numpy as np,json,time
from env_client import make_env
from kinematics_candidate import planar_ik,forward

e=make_env();o,_=e.reset(seed=0)
print('maxsteps',e.max_steps,flush=True)
base=np.array([o[16]-.65,o[17]-.001,0.]);init=o[16:19].copy();start=time.time()
steps=0
rows=[]
for z in np.arange(.60,-.221,-.02):
 q=planar_ik(.65,z,mount_height=0.)
 for k in range(25 if steps==0 else 7):
  a=np.zeros(11,np.float32);a[:3]=np.clip((base-o[93:96])*1.5,-.1,.1);a[3:10]=np.clip((q-o[96:103])*1.5,-.1,.1);a[10]=1.
  o,r,t,tr,info=e.step(a);steps+=1
  if t or tr:break
 p,R=forward(o[96:103],base=o[93:96],mount_height=0.)
 row={'z':float(z),'steps':steps,'q':o[96:103].tolist(),'base':o[93:96].tolist(),'pred':p.tolist(),'obj':o[16:19].tolist(),'dobj':(o[16:19]-init).tolist(),'error':float(np.linalg.norm(q-o[96:103]))}
 rows.append(row)
 print(json.dumps(row),flush=True)
 if t or tr:break
with open('calibration_contact.json','w') as f:json.dump(rows,f)
with open('calibration_state.json','w') as f:json.dump(o.tolist(),f)
e.close();print('elapsed',time.time()-start,flush=True)
