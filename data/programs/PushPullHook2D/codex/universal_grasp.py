from env_client import make_env
import numpy as np, math
def cl(x,m): return max(-m,min(m,x))
def test(seed):
 e=make_env(); o,_=e.reset(seed=seed); init=o.copy(); h0=o[9:11].copy(); th=o[11];u=np.array([math.cos(th),math.sin(th)])
 yc=.65; s=(o[10]-yc)/u[1]; s=np.clip(s,.12,o[18]-.12); p=o[9:11]-s*u
 # retract + travel to bottom center x, rotate upward; horizontal first can collide with arm, retract fixed min
 goals=[np.array([o[0],.15]),np.array([p[0],.15])]
 for goal in goals:
  for k in range(100):
   da=(math.pi/2-o[2]+math.pi)%(2*math.pi)-math.pi
   old=o.copy();o,*_=e.step(np.array([cl(goal[0]-o[0],.05),cl(goal[1]-o[1],.05),cl(da,.196),-.1,0],np.float32))
   if np.linalg.norm(o[:2]-goal)<.01 and abs(da)<.02: break
 # move up with suction until hook changes
 caught=False
 for k in range(40):
  oldh=o[9:11].copy();o,*_=e.step(np.array([0,.02,0,0,1],np.float32))
  if np.linalg.norm(o[9:11]-oldh)>.005: caught=True; break
 # Pull back down: attached iff hook follows
 old=o.copy()
 for k in range(2): o,*_=e.step(np.array([0,-.02,0,0,1],np.float32))
 follow=np.linalg.norm((o[9:11]-old[9:11])-(o[:2]-old[:2]))<.01 and np.linalg.norm(o[9:11]-old[9:11])>.02
 out=(caught,follow,o[:2].copy(),p.copy(),np.linalg.norm(o[9:11]-h0),k)
 e.close();return out
for s in range(50):
 c,f,r,p,m,k=test(s); print(s,int(c),int(f),'rob',np.round(r,2),'p',np.round(p,2),'hm',round(m,2))
