from env_client import make_env
import numpy as np, sys, math, json
ang=float(sys.argv[1]); off=float(sys.argv[2]); seed=int(sys.argv[3]); tag=sys.argv[4]
env=make_env(); obs,info=env.reset(seed=seed)
R=obs.get_object_from_name("robot"); C=obs.get_object_from_name("obstacle_chair")
def S(): return (obs.get(R,"pos_base_x"),obs.get(R,"pos_base_y"),obs.get(C,"x"),obs.get(C,"y"))
ux,uy=math.cos(ang),math.sin(ang); px,py=-uy,ux
bx,by,cx0,cy0=S()
tx,ty=cx0-1.25*ux+off*px, cy0-1.25*uy+off*py
traj=[]; bad=None
for i in range(150):
    bx,by,cx,cy=S(); dx,dy=tx-bx,ty-by; n=math.hypot(dx,dy)
    if n<0.04: break
    a=np.zeros(11,dtype=np.float32); a[0]=min(0.1,n)*dx/n; a[1]=min(0.1,n)*dy/n
    obs,r,te,tr,inf=env.step(a)
    if float(r)!=-1.0: bad=(i,float(r),S()); break
    if te or tr: break
else: pass
if bad is None:
  for i in range(400):
    a=np.zeros(11,dtype=np.float32); a[0]=0.035*ux; a[1]=0.035*uy
    obs,r,te,tr,inf=env.step(a)
    bx,by,cx,cy=S(); traj.append((round(cx,4),round(cy,4)))
    if float(r)!=-1.0: bad=(i,float(r),(round(cx,3),round(cy,3))); break
    if te or tr: break
json.dump(traj,open("probe_rew_traj_%s.json"%tag,"w"))
print(tag,"ang",round(ang,2),"off",off,"seed",seed,"NONMINUS1" if bad else "all-1", bad if bad else "", "n",len(traj),
      "chair_end",traj[-1] if traj else None)
env.close()
