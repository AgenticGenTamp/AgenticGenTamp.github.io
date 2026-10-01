from env_client import make_env
import numpy as np, math

def cl(x,m): return max(-m,min(m,x))
for off in np.arange(.15,.61,.025):
 e=make_env(); o,_=e.reset(seed=0); start=o.copy(); th=o[11]; u=np.array([math.cos(th),math.sin(th)])
 tip=o[9:11]-o[18]*u; goal=tip-off*u
 for k in range(80):
  da=(th-o[2]+math.pi)%(2*math.pi)-math.pi
  a=[cl(goal[0]-o[0],.05),cl(goal[1]-o[1],.05),cl(da,.196),-.1,0]
  o,*_=e.step(np.array(a,np.float32))
  if np.linalg.norm(o[:2]-goal)<.005 and abs(da)<.01: break
 before=o.copy(); o,*_=e.step(np.array([0,0,0,0,1],np.float32))
 # pull directly away after switching suction on
 perp=-u*.03
 o,*_=e.step(np.array([perp[0],perp[1],0,0,1],np.float32))
 hm=np.linalg.norm(o[9:11]-before[9:11]); print(round(off,3),np.round(o[:2],3),round(hm,4))
 e.close()
