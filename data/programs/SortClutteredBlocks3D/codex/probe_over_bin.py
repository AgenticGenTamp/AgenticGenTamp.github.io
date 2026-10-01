"""Test the visually calibrated short-reach pose over the blocking bin."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np

env=make_env(); s,info=env.reset(seed=0,options={"object_count":4})
a=GeneratedApproach(env.action_space,env.observation_space,{}); a.reset(s,info)
for _ in range(166): s,_,_,_,_=env.step(a.get_action(s))
cube=s.get_object_from_name("cube4"); cx,cy=s.get(cube,"x"),s.get(cube,"y")
HOME=np.array([0.,-.349,np.pi,-2.548,0.,-.873,np.pi/2]); HIGH=np.array([0.,1.,np.pi,-1.,0.,1.,0.]); LOW=HIGH.copy();LOW[1]=1.34
def move(bg,qg,n,lim=.1):
 global s,reward
 for _ in range(n):
  r=s.get_object_from_name("robot");b=np.array([s.get(r,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")]);q=np.array([s.get(r,f"pos_arm_joint{i}") for i in range(1,8)])
  x=np.zeros(11,np.float32);e=bg-b;e[2]=(e[2]+np.pi)%(2*np.pi)-np.pi;x[:3]=np.clip(e,-lim,lim);x[3:10]=np.clip(qg-q,-.1,.1);x[10]=0.;s,reward,_,_,_=env.step(x)
move(np.array([-1.,.6,0.]),HOME,80);move(np.array([cx,.60,-np.pi/2]),HOME,40)
move(np.array([cx,cy+.47,-np.pi/2]),HIGH,80);move(np.array([cx,cy+.47,-np.pi/2]),LOW,15)
move(np.array([cx,cy+.27,-np.pi/2]),LOW,25,.012)
def pos(n):
 o=s.get_object_from_name(n);return tuple(round(s.get(o,f),3) for f in ("x","y","z"))
print(reward,pos("cube4"),pos("bin_yellow"),pos("bin_blue"))
env.close()
