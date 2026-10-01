from env_client import make_env
from kin import *
import sys
exec(open('probe3.py').read().split('env=make_env()')[0])
yaw=float(sys.argv[1]); dx=float(sys.argv[2]); dy=float(sys.argv[3])
env=make_env()
x,y=0.648+dx,-0.173+dy
res=[]
for zt in np.arange(0.155,0.30,0.01):
    obs,_=env.reset(seed=0)
    q=getq(obs)
    for z in [0.3, zt]:
        qt,e=ik(q,np.array([x,y,z]),down_R(yaw)); obs,ok=step_to(env,obs,qt); q=getq(obs)
    r=obs.get_object_from_name('robot')
    a=np.zeros(11,dtype=np.float32); a[10]=-1
    obs,*_=env.step(a)
    res.append((round(zt,3), round(fk(q)[2,3],3), obs.get(r,'grasp_active')))
print(yaw,dx,dy,[t for t in res if t[2]>0], 'reached', [t[1] for t in res][:3])
env.close()
