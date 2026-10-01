from env_client import make_env
from kin import *
exec(open('probe3.py').read().split('env=make_env()')[0])
env=make_env()
x,y=0.648,-0.173
for zt in [0.16,0.18,0.20,0.22,0.25]:
    obs,_=env.reset(seed=0)
    q=getq(obs); z=0.25
    while z>=zt-1e-9:
        qt,e=ik(q,np.array([x,y,z]),down_R(np.pi/2)); obs,ok=step_to(env,obs,qt); q=getq(obs); z-=0.01
    r=obs.get_object_from_name('robot'); c=obs.get_object_from_name('cube0')
    a=np.zeros(11,dtype=np.float32); a[10]=-1
    obs,*_=env.step(a)
    print(zt, 'fs',obs.get(r,'finger_state'),'ga',obs.get(r,'grasp_active'),obs.get(c,'grasp_active'), [round(obs.get(r,f),3) for f in ['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']])
    if obs.get(r,'grasp_active')>0.5:
        qt,e=ik(q,np.array([x,y,z+0.2]),down_R(np.pi/2)); obs,ok=step_to(env,obs,qt,grip=0); 
        obs,rew,term,trunc,info=env.step(np.zeros(11,dtype=np.float32))
        print(' lifted cube pose',[round(obs.get(c,f),4) for f in ['pose_x','pose_y','pose_z','pose_qx','pose_qy','pose_qz','pose_qw']],'fk',fk(getq(obs))[:3,3],term,rew)
env.close()
