from probe_lib import *
import sys
seed=int(sys.argv[1]); tipz=float(sys.argv[2]); lead=int(sys.argv[3]) if len(sys.argv)>3 else 70
pr=P(seed); H=[]; TR=float(sys.argv[4]) if len(sys.argv)>4 else 0.012
for i in range(60):
    pr.step(np.zeros(11)); H.append(pr.obs[16:18].copy())
h=np.array(H); x,y=h[:,0],h[:,1]
sol=np.linalg.lstsq(np.c_[x,y,np.ones(len(x))],-(x*x+y*y),rcond=None)[0]
c=-sol[:2]/2; r=np.sqrt(c@c-sol[2])
th=np.unwrap(np.arctan2(y-c[1],x-c[0])); w=(th[-1]-th[-30])/29
thT=th[-1]+w*lead; Pt=c+r*np.array([np.cos(thT),np.sin(thT)])
print('c',c.round(3),'r',round(r,3),'w',round(w,4),'P',Pt.round(3))
psi=thT
pr.goto(pr.ik([Pt[0],Pt[1],0.56],psi)); pr.goto(pr.ik([Pt[0],Pt[1],tipz],psi))
print('arrived t=',len(H)+0,'tip',pr.fk().round(3),'drink',pr.obs[16:19].round(3))
best=9; closed=False
for i in range(250):
    d=np.linalg.norm(pr.obs[16:18]-pr.fk()[:2])
    if not closed and (d<TR or (best<0.02 and d>best+0.002)):
        closed=True; print('close at d',round(d,4),'best',round(best,4))
        for k in range(5): pr.step(np.r_[np.zeros(10),1])
        break
    best=min(best,d)
    pr.step(np.zeros(11))
pr.grip=1
tp=pr.fk(); pr.goto(pr.ik([tp[0],tp[1],tipz+0.1],psi))
print('after lift drink',pr.obs[16:23].round(3),'tip',pr.fk().round(3))
for k in range(20): pr.step(np.r_[np.zeros(10),1])
print('later drink',pr.obs[16:23].round(3))
