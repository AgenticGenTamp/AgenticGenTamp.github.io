"""Place red cube laterally, then nudge its bin center onto it."""
from env_client import make_env
import numpy as np

Q = np.array([0.,1.3,np.pi,-1.7,0.,1.,0.])
HOME = np.array([0.,-.349,np.pi,-2.548,0.,-.873,np.pi/2])
env=make_env(); s,_=env.reset(seed=0,options={"object_count":4})
def step_to(bg,qg,n,speed=.1):
    global s, reward, term
    for _ in range(n):
        r=s.get_object_from_name("robot")
        b=np.array([s.get(r,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")])
        q=np.array([s.get(r,f"pos_arm_joint{i}") for i in range(1,8)])
        a=np.zeros(11,np.float32); a[:3]=np.clip(.9*(bg-b),-speed,speed)
        a[3:10]=np.clip(.9*(qg-q),-.1,.1); a[10]=0.
        s,reward,term,trunc,_=env.step(a)

# Reach first contact with the low-y exposed red cube.
step_to(np.array([1.,.55,np.pi]),HOME,35)
step_to(np.array([-1.,.55,0.]),HOME,35)
step_to(np.array([-1.,.016,0.]),Q,80)
step_to(np.array([-.906,.016,0.]),Q,12,.012)
qs=Q.copy(); qs[0]=.8
step_to(np.array([-.906,.016,0.]),qs,58)
# Disengage, fold, and approach red bin from its outer (-x) side.
step_to(np.array([-1.08,.016,0.]),qs,12,.03)
step_to(np.array([-1.08,-.15,0.]),HOME,80)
step_to(np.array([-1.08,-.15,0.]),Q,80)
step_to(np.array([-.91,-.15,0.]),Q,18,.012)
def xyz(name):
    o=s.get_object_from_name(name); return np.array([s.get(o,f) for f in ("x","y","z")])
print(reward,term,"cube",np.round(xyz("cube1"),4),"bin",np.round(xyz("bin_red"),4),"dist",round(float(np.linalg.norm(xyz("cube1")-xyz("bin_red"))),4))
env.close()
