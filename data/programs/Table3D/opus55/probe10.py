from env_client import make_env
from kin import *
import sys
exec(open('probe3.py').read().split('env=make_env()')[0])
yaw=float(sys.argv[1]); zt=float(sys.argv[2])
env=make_env()
hits=[];stuck=[]
for dx in np.arange(-0.03,0.031,0.01):
  for dy in np.arange(-0.03,0.031,0.01):
    x,y=0.648+dx,-0.173+dy
    obs,_=env.reset(seed=0); q=getq(obs)
    for z in [0.3,0.22,zt]:
        qt,e=ik(q,np.array([x,y,z]),down_R(yaw)); obs,ok=step_to(env,obs,qt); q=getq(obs)
    if abs(fk(q)[2,3]-zt)>0.003: stuck.append((round(dx,2),round(dy,2),round(fk(q)[2,3],3))); continue
    r=obs.get_object_from_name('robot')
    a=np.zeros(11,dtype=np.float32); a[10]=-1
    obs,*_=env.step(a)
    if obs.get(r,'grasp_active')>0: hits.append((round(dx,2),round(dy,2)))
print('yaw',yaw,'z',zt,'hits',hits,'stuck',stuck)
env.close()
