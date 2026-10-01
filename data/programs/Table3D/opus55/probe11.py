from env_client import make_env
from kin import *
import sys
exec(open('probe3.py').read().split('env=make_env()')[0])
env=make_env()
out=[]
for yaw in [0, np.pi/2]:
  for (dx,dy) in [(0,0),(0.04,0),(0,0.04),(-0.04,0),(0,-0.04),(0.02,0),(0,0.02)]:
    x,y=0.648+dx,-0.173+dy
    obs,_=env.reset(seed=0); q=getq(obs); z=0.3
    qt,e=ik(q,np.array([x,y,z]),down_R(yaw)); obs,ok=step_to(env,obs,qt); q=getq(obs)
    while z>0.1:
        qt,e=ik(q,np.array([x,y,z-0.004]),down_R(yaw)); obs,ok=step_to(env,obs,qt)
        if not ok: break
        q=getq(obs); z-=0.004
    out.append((round(yaw,2),dx,dy,round(fk(q)[2,3],3)))
print(out)
env.close()
