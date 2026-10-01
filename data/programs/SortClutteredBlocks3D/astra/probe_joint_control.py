from env_client import make_env
import numpy as np
for mode in ['constant','feedback']:
 e=make_env(); s,_=e.reset(seed=0); r=s.get_object_from_name('robot'); q0=s.get(r,'pos_arm_joint7'); prev=q0; delta=0
 print(mode,flush=True)
 for k in range(12):
  a=np.zeros(11)
  if mode=='constant': a[9]=.1
  else: a[9]=np.clip((q0+.2-prev)/.4+1.5*delta,-.1,.1)
  s,rew,done,trunc,info=e.step(a);q=s.get(r,'pos_arm_joint7');delta=q-prev;prev=q
  print(k,round(a[9],5),round(q-q0,5),round(s.get(r,'vel_arm_joint7'),5),flush=True)
 e.close()
