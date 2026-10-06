import time, numpy as np
from planner import *
tp=TwinPlanner(4); s,_=tp.env.reset(seed=0); tp.state=s
e=tp.ee()
t=time.time()
for i in range(50):
    q=tp.ik(Pose((0.3+0.001*i,0.0,0.2),tp.down))
print("ik (with snapshot)",(time.time()-t)/50)
t=time.time()
for i in range(50):
    q=pybullet_inverse_kinematics(tp.arm, Pose((0.3+0.001*i,0.0,0.2),tp.down), validate=True)
print("ik raw",(time.time()-t)/50)
env=tp.env
t=time.time()
for i in range(100):
    env._set_robot_and_held_object(env.robot.get_base(), list(q[:7]), tp.arm.get_finger_state())
    c=env._robot_or_held_object_collision_exists()
print("coll check",(time.time()-t)/100, c)
