from env_client import make_env
import numpy as np, sys, math
ang=float(sys.argv[1]); mag=float(sys.argv[2]); seed=int(sys.argv[3])
env=make_env(); obs,info=env.reset(seed=seed)
R=obs.get_object_from_name("robot"); C=obs.get_object_from_name("obstacle_chair")
def S(): return (obs.get(R,"pos_base_x"),obs.get(R,"pos_base_y"),obs.get(C,"x"),obs.get(C,"y"))
rs=set()
# 1) go to a point on the opposite side of the chair from push direction
ux,uy=math.cos(ang),math.sin(ang)
bx,by,cx,cy=S(); tx,ty=cx-1.3*ux,cy-1.3*uy
for i in range(200):
    bx,by,cx,cy=S(); dx,dy=tx-bx,ty-by; n=math.hypot(dx,dy)
    if n<0.05: break
    a=np.zeros(11,dtype=np.float32); a[0]=min(0.1,n)*dx/n; a[1]=min(0.1,n)*dy/n
    obs,r,te,tr,inf=env.step(a); rs.add(float(r))
    if te or tr: print("term during approach",i); env.close(); sys.exit()
# 2) push slowly along ang
maxd=0
for i in range(600):
    a=np.zeros(11,dtype=np.float32); a[0]=mag*ux; a[1]=mag*uy
    obs,r,te,tr,inf=env.step(a); rs.add(float(r))
    bx,by,cx,cy=S()
    d=math.hypot(cx-(0),cy-0)
    if abs(float(r)+1.0)>1e-9:
        print("NON-1 ang",round(ang,2),"r",r,"chair",round(cx,3),round(cy,3)); break
    if te or tr:
        print("TERM ang",round(ang,2),"mag",mag,"step",i,"chair",round(cx,3),round(cy,3),"base",round(bx,3),round(by,3)); break
else:
    print("done ang",round(ang,2),"mag",mag,"chair",round(cx,3),round(cy,3))
print("  rewards",rs)
env.close()
