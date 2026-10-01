from env_client import make_env
import numpy as np
env=make_env(); obs,info=env.reset(seed=1)
R=obs.get_object_from_name("robot")
rs=set()
def go(a,n,tag):
    global obs
    for i in range(n):
        obs,r,te,tr,inf=env.step(np.array(a,dtype=np.float32))
        rs.add(round(float(r),6))
        if abs(r+1.0)>1e-9: print("NONE-1!",tag,i,r,inf); return True
        if te or tr: print("term",tag,i,te,tr); return True
    return False
# 1. arm joint sweeps both signs
for j in range(7):
    for s in (1,-1):
        a=[0]*11; a[3+j]=0.1*s
        if go(a,40,f"j{j+1}s{s}"): break
# gripper
for g in (1.0,0.0,1.0):
    a=[0]*11; a[10]=g; go(a,10,f"grip{g}")
# 2. yaw sweep
for s in (1,-1):
    a=[0]*11; a[2]=0.1*s; go(a,80,f"yaw{s}")
# 3. far base positions
for d in [(1,0),(0,1),(-1,0),(0,-1)]:
    a=[0.1*d[0],0.1*d[1],0,0,0,0,0,0,0,0,0]; go(a,300,f"far{d}")
p=obs.get_object_from_name("robot")
print("final base",obs.get(p,"pos_base_x"),obs.get(p,"pos_base_y"))
print("distinct rewards:",rs)
env.close()
