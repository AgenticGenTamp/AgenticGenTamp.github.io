from env_client import make_env
import numpy as np
env=make_env(); obs,info=env.reset(seed=1)
R=obs.get_object_from_name("robot"); C=obs.get_object_from_name("obstacle_chair")
def st(): 
    return (obs.get(R,"pos_base_x"),obs.get(R,"pos_base_y"),obs.get(C,"x"),obs.get(C,"y"))
print("start",[round(v,3) for v in st()])
rs={}
for i in range(400):
    bx,by,cx,cy=st()
    dx,dy=cx-bx,cy-by
    n=(dx*dx+dy*dy)**0.5
    # push chair in +x direction: approach from -x side then push
    a=np.zeros(11,dtype=np.float32)
    a[0]=0.1*dx/max(n,1e-6); a[1]=0.1*dy/max(n,1e-6)
    obs,r,te,tr,inf=env.step(a)
    rr=round(float(r),6); rs[rr]=rs.get(rr,0)+1
    if abs(r+1.0)>1e-9: print("NON -1 at step",i,r,[round(v,3) for v in st()]); break
    if i%25==0: print(i,"r",r,[round(v,3) for v in st()])
    if te or tr: print("TERM step",i,te,tr,"r",r,[round(v,3) for v in st()]); break
print("rewards",rs)
env.close()
