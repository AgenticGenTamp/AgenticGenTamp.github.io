from env_client import make_env
from kin import *
exec(open('probe3.py').read().split('env=make_env()')[0])
env=make_env()
x,y=0.648-0.12,-0.173
obs,_=env.reset(seed=0); q=getq(obs)
for z in [0.3,0.22,0.19]:
    qt,e=ik(q,np.array([x,y,z]),down_R(np.pi/2)); obs,ok=step_to(env,obs,qt); q=getq(obs)
r=obs.get_object_from_name('robot'); c=obs.get_object_from_name('cube0')
a=np.zeros(11,dtype=np.float32); a[10]=-1
obs,rw,te,tr,_=env.step(a)
print('grasp',obs.get(r,'grasp_active'),obs.get(c,'grasp_active'),obs.get(r,'finger_state'),[round(obs.get(r,f),4) for f in ['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']])
for z in [0.21,0.23,0.25,0.27,0.29,0.31,0.33]:
    qt,e=ik(q,np.array([x,y,z]),down_R(np.pi/2)); 
    a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qt-q,-.4,.4)
    obs,rw,te,tr,_=env.step(a); q=getq(obs)
    T=fk(q)
    print(z,'cube',[round(obs.get(c,f),4) for f in ['pose_x','pose_y','pose_z','pose_qx','pose_qy','pose_qz','pose_qw']],'flange',T[:3,3].round(4),rw,te,tr)
    if te: break
env.close()
