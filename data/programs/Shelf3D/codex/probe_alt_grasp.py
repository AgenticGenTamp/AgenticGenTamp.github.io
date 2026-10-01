"""Live grasp test for the lowest stable alternate q3/q5 branch."""
import numpy as np
from env_client import make_env

TARGET=np.array([-.12,2.145,-2.847,0.0,-.295,-.955,-.267])
JF=[f"pos_arm_joint{i}" for i in range(1,8)]

def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def q(s):
    r=s.get_object_from_name("robot")
    return np.array([s.get(r,f) for f in JF],float)

env=make_env();state,_=env.reset(seed=0,options={"object_count":1})
cube0=np.array([g(state,"cube1",f) for f in ("x","y","z")])
for step in range(390):
    a=np.zeros(11,np.float32); cur=q(state)
    active=(0,2,4,5,6) if step<150 else range(7)
    for j in active:a[3+j]=np.clip(2*(TARGET[j]-cur[j]),-.1,.1)
    if step>=150:
        a[0]=np.clip(2*(.38-g(state,"robot","pos_base_x")),-.1,.1)
        a[1]=np.clip(2*(cube0[1]-g(state,"robot","pos_base_y")),-.1,.1)
    state,_,_,_,_=env.step(a)
print("settled",np.round(q(state),3).tolist(),"base",round(g(state,"robot","pos_base_x"),3))
hit=None
for step in range(80):
    a=np.zeros(11,np.float32);cur=q(state)
    for j in range(7):a[3+j]=np.clip(2*(TARGET[j]-cur[j]),-.1,.1)
    # Small raster around visually inferred horizontal alignment.
    a[0]=.012 if (step//10)%2==0 else -.012
    a[1]=.008 if (step//20)%2==0 else -.008
    a[10]=0.0 if step%10<4 else 1.0
    state,_,_,_,_=env.step(a)
    cube=np.array([g(state,"cube1",f) for f in ("x","y","z")])
    if np.linalg.norm(cube-cube0)>.002:
        hit=(step,cube.copy(),g(state,"robot","pos_base_x"),g(state,"robot","pos_base_y"));break
print("HIT" if hit else "NO",hit)
env.close()
