import sys, numpy as np
from helper import Sim, wrap, R
from expA_lib import rel, robot_for
sd=int(sys.argv[1]) if len(sys.argv)>1 else 42
inc=float(sys.argv[2]) if len(sys.argv)>2 else 0.01
gap=float(sys.argv[3]) if len(sys.argv)>3 else 0.12   # initial centerline->button offset
s=Sim(sd)
s.grasp_hook(1.15)
o=s.obs
print("after grasp: R=",np.round(o[:3],4),"hook=",np.round(o[9:12],4))
Cl,dth=rel(s)
print("rel Cl=",np.round(Cl,4),"dth=",round(dth,4))
M=o[20:22].copy()
# want hth=pi/2 : bar from C going down. normal nh=(-sin,cos)=(-1,0). push +x => bar at x=M.x-gap
hth=np.pi/2
Cdes=np.array([M[0]-gap, M[1]+0.30])   # corner 0.30 above button y, bar covers down 1.25
Rp,rth=robot_for(Cl,dth,Cdes,hth)
print("target robot",np.round(Rp,4),round(rth,4))
k=s.goto(o[0],o[1],rth,0.2,1.0); print("rot",k,"hth now",round(s.obs[11],4))
k=s.goto(Rp[0],s.obs[1],rth,0.2,1.0); print("mx",k)
k=s.goto(Rp[0],Rp[1],rth,0.2,1.0); print("mxy",k)
o=s.obs
print("placed: hook=",np.round(o[9:12],4)," M=",np.round(o[20:22],4))
cl=o[9]  # centerline x  (hth=pi/2 -> bar vertical at x=C.x)
print("centerline x=%.4f  M.x=%.4f  d=%.4f"%(cl,o[20],o[20]-cl))
print("--- push +x by inc=%g ---"%inc)
M0=o[20:22].copy(); C0=o[9:11].copy()
rows=[]
for i in range(40):
    p=s.obs[[0,1,9,10,20,21]].copy()
    s.step([inc,0,0,0,1.0])
    o=s.obs
    dC=o[9]-p[2]; dM=o[20]-p[4]; dMy=o[21]-p[5]
    rows.append((i,o[9],o[20]-o[9],dC,dM,dMy,s.term))
    if s.term: break
for r in rows[:44]:
    print("i=%2d Cx=%.4f sep=%.4f dCx=%+.4f dMx=%+.4f dMy=%+.5f term=%s"%r)
o=s.obs
print("TOT dC=%.4f dM=(%.4f,%.4f) |MT|=%.4f term=%s steps=%d"%(o[9]-C0[0],o[20]-M0[0],o[21]-M0[1],np.hypot(o[20]-o[29],o[21]-o[30]),s.term,s.n))
s.close()
