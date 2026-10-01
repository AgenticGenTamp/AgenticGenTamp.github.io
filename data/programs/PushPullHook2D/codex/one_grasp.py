from env_client import make_env
import numpy as np, math
def cl(x,m): return max(-m,min(m,x))
e=make_env(); o,_=e.reset(seed=0); th=o[11];u=np.array([math.cos(th),math.sin(th)]); tip=o[9:11]-o[18]*u;p=tip+.3*u; goal=p-np.array([0,.35]); print('u tip p goal',u,tip,p,goal)
for k in range(100):
 da=(math.pi/2-o[2]+math.pi)%(2*math.pi)-math.pi
 old=o.copy();o,*_=e.step(np.array([cl(goal[0]-o[0],.05),cl(goal[1]-o[1],.05),cl(da,.196),-.1,1],np.float32))
 if k%5==0 or np.linalg.norm(o[:2]-old[:2])<.001: print(k,np.round(o[[0,1,2,4,6,9,10,11]],3), 'd',np.round(o[[0,1,9,10]]-old[[0,1,9,10]],3))
 if np.linalg.norm(o[:2]-goal)<.008 and abs(da)<.02: break
print('pull')
for k in range(5):
 old=o.copy();o,*_=e.step(np.array([0,-.03,0,0,1],np.float32));print(k,np.round(o[[0,1,2,9,10,11]],3),'d',np.round(o[[0,1,9,10]]-old[[0,1,9,10]],3))
e.close()
