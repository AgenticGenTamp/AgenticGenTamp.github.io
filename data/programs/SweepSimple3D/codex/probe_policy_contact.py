"""Trace current policy to identify when and how its wiper contact is lost."""
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def vec(s,n,fs): return np.array([v(s,n,f) for f in fs])


e=make_env(); s,info=e.reset(seed=0,options={"object_count":5})
p=GeneratedApproach(e.action_space,e.observation_space,{}); p.reset(s,info)
w0=vec(s,"wiper_0","xyz")
for k in range(125):
    before=vec(s,"wiper_0","xyz")
    a=p.get_action(s)
    s,r,t,tr,_=e.step(a)
    w=vec(s,"wiper_0","xyz")
    if k%3==0 or np.linalg.norm(w-before)>.01:
      print(k+1,"base",vec(s,"robot",("pos_base_x","pos_base_y","pos_base_rot")).round(3).tolist(),
            "q",vec(s,"robot",tuple("pos_arm_joint%d"%i for i in range(1,8))).round(2).tolist(),
            "g",round(v(s,"robot","pos_gripper"),2),"act",a.round(2).tolist(),
            "w",w.round(3).tolist(),"dw",(w-before).round(3).tolist(),
            "wquat",vec(s,"wiper_0",("qw","qx","qy","qz")).round(2).tolist())
    if t or tr: break
e.close()
