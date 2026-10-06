import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
import approach as A
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,info=env.reset(seed=0)
ap=A.GeneratedApproach(env.action_space,env.observation_space,{})
ap.reset(obs,info)
# custom plan: go to pregrasp then descend, logging
base_t=np.array([1.4,-0.15,np.pi]); R=A.R_down(0.3)
g=np.array([0.955,-0.385,0.50])
def plan():
    yield from ap.pose_move(g+[0,0,0.08],R,base_t,max_steps=200)
    print('pre done', ap.ee()[0].round(3), 'q',ap.q().round(2))
    yield from ap.cart_move(g,R,speed=0.006,base_t=base_t,max_steps=60)
ap.gen=plan()
for i in range(300):
    a=ap.get_action(obs)
    obs,*_=env.step(a)
    if i>40 and i%3==0: print(i, ap.ee()[0].round(3), 'a',a[3:10].round(3))
