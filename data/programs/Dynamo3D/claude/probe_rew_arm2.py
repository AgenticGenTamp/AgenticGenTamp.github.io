from env_client import make_env
import numpy as np
env=make_env(); obs,info=env.reset(seed=11)
R=obs.get_object_from_name("robot"); C=obs.get_object_from_name("obstacle_chair")
def s():
    return dict(bx=obs.get(R,"pos_base_x"),by=obs.get(R,"pos_base_y"),cx=obs.get(C,"x"),cy=obs.get(C,"y"),cz=obs.get(C,"z"))
rs={}
def step(a,n,tag):
    global obs
    for i in range(n):
        obs,r,te,tr,inf=env.step(np.array(a,dtype=np.float32))
        rr=repr(float(r)); rs[rr]=rs.get(rr,0)+1
        if abs(float(r)+1.0)>1e-9: print("NON-1",tag,i,r,{k:round(v,3) for k,v in s().items()}); return "R"
        if te or tr: print("TERM",tag,i,{k:round(v,3) for k,v in s().items()}); return "T"
    return None
# approach chair to ~0.75m then use arm to touch it
st=s(); dx=st['cx']-st['bx']; dy=st['cy']-st['by']; n=(dx*dx+dy*dy)**.5
steps=int((n-0.85)/0.087)
print("approach steps",steps,"dist",round(n,2))
a=[0]*11; a[0]=0.1*dx/n; a[1]=0.1*dy/n
print(step(a,max(steps,0),"approach"), {k:round(v,3) for k,v in s().items()})
# open gripper, wave arm joints toward chair
for j,sgn,cnt in [(1,1,30),(1,-1,60),(3,1,30),(3,-1,60),(0,1,60),(0,-1,120),(5,1,40),(5,-1,80)]:
    a=[0]*11; a[3+j]=0.1*sgn; a[10]=1.0
    res=step(a,cnt,f"arm j{j+1} {sgn}")
    if res: break
print("chair",{k:round(v,3) for k,v in s().items()})
print("rewards",rs)
env.close()
