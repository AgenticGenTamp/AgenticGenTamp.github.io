from env_client import make_env
import numpy as np, math

def clip(x,m): return max(-m,min(m,x))
def test(seed, off=.23):
 e=make_env(); o,_=e.reset(seed=seed); init=o.copy(); ht=o[11]; u=np.array([math.cos(ht),math.sin(ht)])
 tip=o[9:11]-o[18]*u; base=tip-off*u
 # retract, orient toward hook along +u, move simultaneously
 for k in range(100):
  da=(ht-o[2]+math.pi)%(2*math.pi)-math.pi
  a=np.array([clip(base[0]-o[0],.05),clip(base[1]-o[1],.05),clip(da,.196),-.1,0],np.float32)
  o,r,t,tr,i=e.step(a)
  if np.linalg.norm(o[:2]-base)<.012 and abs(da)<.03: break
 # vacuum and small wiggle/pull
 for a in ([0,0,0,0,1],[.02,0,0,0,1]): o,r,t,tr,i=e.step(np.array(a,np.float32))
 moved=np.linalg.norm(o[9:11]-init[9:11])
 e.close(); return init, o, base, tip, moved

for s in range(30):
 a,o,b,t,m=test(s)
 print(s,'tip',np.round(t,2),'base',np.round(b,2),'actual',np.round(o[:2],2),'hookmove',round(m,3),'vac',o[6])
