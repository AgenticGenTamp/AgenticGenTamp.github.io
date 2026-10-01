from env_client import make_env
import numpy as np

env=make_env(); s,info=env.reset(seed=0)
t=env.observation_space.get_type('mujoco_tidybot_robot'); robot=s.get_objects(t)[0]
features=['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(i) for i in range(1,8)]+['pos_gripper']
objtype=env.observation_space.get_type('mujoco_movable_object')
def vals(s): return [round(s.get(robot,f),4) for f in features]
print('INITIAL',vals(s),flush=True)
for label,index,value,steps in [('close',10,0,4),('open',10,1,4),('base x',0,0.1,4),('base yaw',2,0.1,4),('joint1',3,0.1,4),('joint2',4,0.1,4),('joint6',8,0.1,4),('zero',0,0,4)]:
    for k in range(steps):
        a=np.zeros(11); a[10]=1; a[index]=value
        s,r,te,tr,info=env.step(a)
        print(label,k,vals(s),r,info,flush=True)
env.close()
env=make_env(); s,info=env.reset(seed=1); robot=s.get_objects(t)[0]
for label,index,value,steps in [('yaw',2,.1,16),('x world',0,.1,3),('y world',1,.1,3),('large joint',4,-.1,12),('half gripper',10,.5,1),('close',10,0,1)]:
    for k in range(steps):
        a=np.zeros(11); a[index]=value
        s,r,te,tr,info=env.step(a)
        if k==steps-1: print('ROUND2',label,vals(s),r,flush=True)
env.close()
