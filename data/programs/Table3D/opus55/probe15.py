from env_client import make_env
from kin import *
import sys
exec(open('probe3.py').read().split('env=make_env()')[0])
env=make_env(); yaw=float(sys.argv[1]); res=[]
for ox in np.arange(0.08,0.18,0.01):
  for zt in [0.16,0.18,0.20]:
    x,y=0.648-ox,-0.173
    obs,_=env.reset(seed=0); q=getq(obs)
    for z in [0.3,0.22,zt]:
        qt,e=ik(q,np.array([x,y,z]),down_R(yaw)); obs,ok=step_to(env,obs,qt); q=getq(obs)
    r=obs.get_object_from_name('robot')
    a=np.zeros(11,dtype=np.float32); a[10]=-1
    obs,*_=env.step(a)
    res.append((round(ox,2),zt,round(fk(q)[2,3],3),obs.get(r,'grasp_active')))
print(yaw,res)
env.close()
