from env_client import make_env
import numpy as np, math

def cl(x,m): return max(-m,min(m,x))
for along in [0,.1,.2,.3,.4]:
 for reach in [.30,.35,.38,.40,.42,.44]:
  e=make_env(); o,_=e.reset(seed=0); init=o.copy(); th=o[11]; u=np.array([math.cos(th),math.sin(th)])
  tip=o[9:11]-o[18]*u; p=tip+along*u; h=np.array([0.,1.]); goal=p-reach*h
  for k in range(100):
   da=(math.pi/2-o[2]+math.pi)%(2*math.pi)-math.pi
   o,*_=e.step(np.array([cl(goal[0]-o[0],.05),cl(goal[1]-o[1],.05),cl(da,.196),-.1,1],np.float32))
   if np.linalg.norm(o[:2]-goal)<.008 and abs(da)<.02: break
  before=o.copy()
  for k in range(3): o,*_=e.step(np.array([0,-.03,0,0,1],np.float32))
  hm=np.linalg.norm(o[9:11]-before[9:11]); print(along,reach,'at',np.round(o[:2],2),'hm',round(hm,3))
  e.close()
