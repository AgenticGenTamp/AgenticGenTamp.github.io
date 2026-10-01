from env_client import make_env
import numpy as np
E=make_env();s,_=E.reset(seed=42);r=s.get_objects(E.observation_space.get_type('kin_robot'))[0]
def act(a,n,tag):
 global s
 for j in range(n):
  s,_,term,trunc,info=E.step(np.array(a,dtype=float))
  if term:print('SUCCESS',tag,j,flush=True);break
 print(tag,'robot',*[round(s.get(r,f),3) for f in ('x','y','theta')],'small',[(round(s.get(o,'x'),2),round(s.get(o,'y'),2)) for t in ('small_circle','small_square') for o in s.get_objects(E.observation_space.get_type(t))],flush=True)
for a,n,tag in [([0,.03,0,0,0],20,'up'),([0,0,.098,0,0],18,'rotate'),([-.03,0,0,0,0],80,'left'),([0,-.03,0,0,0],90,'down'),([.03,0,0,0,0],100,'sweep'),([0,.015,0,0,0],110,'lift'),([.03,0,0,0,0],50,'over'),([0,0,.098,0,0],16,'tip'),([0,0,0,0,0],40,'wait')]:act(a,n,tag)
E.close()
