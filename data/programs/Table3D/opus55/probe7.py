from env_client import make_env
from kin import *
exec(open('probe3.py').read().split('env=make_env()')[0])
env=make_env()
x,y=0.648,-0.173
def info(obs,tag):
    r=obs.get_object_from_name('robot'); c=obs.get_object_from_name('cube0')
    print(tag,'fs',obs.get(r,'finger_state'),'ga',obs.get(r,'grasp_active'),obs.get(c,'grasp_active'),'cz',round(obs.get(c,'pose_z'),3),'q',getq(obs)[:3])
obs,_=env.reset(seed=0); q=getq(obs)
for z in [0.3,0.2,0.16]:
    qt,e=ik(q,np.array([x,y,z]),down_R(np.pi/2)); obs,ok=step_to(env,obs,qt); q=getq(obs)
info(obs,'at 0.16')
for v in [-1,-1,-1,1,-1]:
    a=np.zeros(11,dtype=np.float32); a[10]=v; obs,*_=env.step(a); info(obs,f'grip {v}')
# lift
qt,e=ik(q,np.array([x,y,0.3]),down_R(np.pi/2)); obs,ok=step_to(env,obs,qt,grip=-1); info(obs,'lift')
# B: close while descending
obs,_=env.reset(seed=0); q=getq(obs); z=0.25
while z>0.14:
    qt,e=ik(q,np.array([x,y,z]),down_R(np.pi/2))
    a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qt-q,-.4,.4); a[10]=-1
    obs,*_=env.step(a); q=getq(obs); z-=0.01
    info(obs,f'desc z={z:.3f} fkz={fk(q)[2,3]:.3f}')
env.close()
