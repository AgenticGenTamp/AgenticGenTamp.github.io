from env_client import make_env
from kin import *
import sys
exec(open('probe3.py').read().split('env=make_env()')[0])
env=make_env()
yaw=float(sys.argv[1]); axis=sys.argv[2]
res=[]
for v in np.arange(float(sys.argv[3]),float(sys.argv[4]),0.02):
    x,y=(v,0.0) if axis=='x' else (0.6, v)
    obs,_=env.reset(seed=0); q=getq(obs); z=0.3
    qt,e=ik(q,np.array([x,y,z]),down_R(yaw)); obs,ok=step_to(env,obs,qt); q=getq(obs)
    while z>0.0:
        qt,e=ik(q,np.array([x,y,z-0.01]),down_R(yaw)); obs,ok=step_to(env,obs,qt)
        if not ok: break
        q=getq(obs); z-=0.01
    res.append((round(v,2),round(fk(q)[2,3],3)))
print(yaw,axis,res)
env.close()
