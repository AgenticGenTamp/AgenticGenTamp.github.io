import sys, numpy as np, kin, ctrl
from env_client import make_env
q0=np.array([0,-0.3491,3.1416,-2.5482,0,-0.8727,1.5708])
rng=np.random.default_rng(0)
def cands(al_deg,DX,zz):
    al=np.radians(al_deg)
    z=np.array([np.cos(al),0,-np.sin(al)]); x=np.array([0,1.,0]); y=np.cross(z,x)
    T=ctrl.pose_from([DX,0,zz-ctrl.MOUNT[2]],np.column_stack([x,y,z]))
    sols=[]
    for k in range(300):
        s=q0+rng.uniform(-2.5,2.5,7) if k else q0.copy()
        qd,ep,er=kin.ik2(T,s,tool_z=ctrl.TOOL,q_lim=ctrl.SOFT,iters=100)
        if ep>0.003 or er>0.03: continue
        for j in (0,2,4,6):
            while qd[j]-q0[j]>np.pi: qd[j]-=2*np.pi
            while qd[j]-q0[j]<-np.pi: qd[j]+=2*np.pi
        if np.any(qd<ctrl.SOFT[0]-1e-6) or np.any(qd>ctrl.SOFT[1]+1e-6): continue
        sols.append((np.max(np.abs(qd-q0)),qd))
    sols.sort(key=lambda t:t[0])
    uniq=[]
    for c,qd in sols:
        if all(np.max(np.abs(qd-u[1]))>0.3 for u in uniq): uniq.append((c,qd))
    return T,uniq
env=make_env()
for al_deg,DX in [(45,0.45),(45,0.50),(60,0.45),(60,0.50),(35,0.50),(45,0.40)]:
    T,uniq=cands(al_deg,DX,0.023)
    print("=== alpha=%d DX=%.2f  n=%d"%(al_deg,DX,len(uniq)))
    for c,qd in uniq[:3]:
        o,i=env.reset(seed=0,options={'object_count':1})
        for t in range(260):
            q,b=ctrl.read_robot(o)
            a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip((qd-q)/0.25,-0.1,0.1)
            o,r,te,tr,inf=env.step(a)
            if np.max(np.abs(qd-q))<0.005: break
        q,b=ctrl.read_robot(o); Tach=ctrl.ee_world(q,b)
        want=ctrl.MOUNT+np.array([DX,0,0.023])
        print("  cost=%.2f steps=%d jerr=%.3f eeerr=%.3f"%(c,t,np.max(np.abs(qd-q)),np.linalg.norm(Tach[:3,3]-np.array([b[0],b[1],0])-want)), np.round(qd,2))
env.close()
