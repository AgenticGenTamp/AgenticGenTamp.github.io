"""Guide one exposed cube to its bin with coordinated base/q1 motion."""
from env_client import make_env
import numpy as np

env=make_env(); s,_=env.reset(seed=0,options={"object_count":4})
Q=np.array([0.,1.3,np.pi,-1.7,0.,1.,0.]); HOME=np.array([0.,-.349,np.pi,-2.548,0.,-.873,np.pi/2])
def command(bg,qg,n,speed=.1):
    global s,reward,term
    for _ in range(n):
        r=s.get_object_from_name("robot")
        b=np.array([s.get(r,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")])
        q=np.array([s.get(r,f"pos_arm_joint{i}") for i in range(1,8)])
        a=np.zeros(11,np.float32); a[:3]=np.clip(bg(b,q)-b,-speed,speed) if callable(bg) else np.clip(bg-b,-speed,speed)
        goalq=qg(q) if callable(qg) else qg; a[3:10]=np.clip(goalq-q,-.1,.1); a[10]=0.
        s,reward,term,_,_=env.step(a)

cube=s.get_object_from_name("cube1"); start=np.array([s.get(cube,"x"),s.get(cube,"y")])
binobj=s.get_object_from_name("bin_red"); target=np.array([s.get(binobj,"x"),s.get(binobj,"y")])
command(np.array([1.,.55,np.pi]),Q,80); command(np.array([-1.,.55,0.]),Q,80)
radius=.873
stage=np.array([start[0]-radius,start[1]+.042,0.])
command(np.array([-1.,start[1]+.042,0.]),Q,70)
command(stage,Q,12,.012)
qfinal=.36
def qgoal(q):
    goal=Q.copy(); goal[0]=qfinal; return goal
def basegoal(b,q):
    frac=min(1.,max(0.,q[0]/qfinal))
    point=start+frac*(target-start)
    return np.array([point[0]-radius*np.cos(q[0]), point[1]+.042+radius*np.sin(q[0]),0.])
command(basegoal,qgoal,85,.04)
def xyz(name):
    o=s.get_object_from_name(name); return np.array([s.get(o,f) for f in ("x","y","z")])
print(reward,term,"cube",np.round(xyz("cube1"),4),"bin",np.round(xyz("bin_red"),4),"dist",round(float(np.linalg.norm(xyz("cube1")-xyz("bin_red"))),4))
env.close()
