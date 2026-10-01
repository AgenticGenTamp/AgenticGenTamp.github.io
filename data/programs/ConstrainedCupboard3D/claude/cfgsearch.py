import numpy as np, kin, ctrl
from env_client import make_env
q0=np.array([0,-0.3491,3.1416,-2.5482,0,-0.8727,1.5708])
al=np.radians(45.); DX=0.50
z=np.array([np.cos(al),0,-np.sin(al)]); x=np.array([0,1.,0]); y=np.cross(z,x)
R=np.column_stack([x,y,z])
T=ctrl.pose_from([DX,0,0.023-ctrl.MOUNT[2]],R)
rng=np.random.default_rng(0)
sols=[]
for k in range(400):
    s=q0+rng.uniform(-2.5,2.5,7) if k else q0.copy()
    qd,ep,er=kin.ik2(T,s,tool_z=ctrl.TOOL,q_lim=ctrl.SOFT,iters=100)
    if ep>0.003 or er>0.03: continue
    for j in (0,2,4,6):
        while qd[j]-q0[j]>np.pi: qd[j]-=2*np.pi
        while qd[j]-q0[j]<-np.pi: qd[j]+=2*np.pi
    if np.any(qd<ctrl.SOFT[0]) or np.any(qd>ctrl.SOFT[1]): continue
    sols.append((np.max(np.abs(qd-q0)),qd))
sols.sort(key=lambda t:t[0])
uniq=[]
for c,qd in sols:
    if all(np.max(np.abs(qd-u[1]))>0.3 for u in uniq): uniq.append((c,qd))
print(len(sols),"uniq",len(uniq))
env=make_env(); o,i=env.reset(seed=0,options={'object_count':1})
for c,qd in uniq[:6]:
    o,i=env.reset(seed=0,options={'object_count':1})
    n=0
    for t in range(300):
        q,b=ctrl.read_robot(o)
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip((qd-q)/0.25,-0.1,0.1)
        o,r,te,tr,inf=env.step(a); n=t
        if np.max(np.abs(qd-q))<0.006: break
    q,b=ctrl.read_robot(o)
    Tach=ctrl.ee_world(q,b)
    print("cost=%.2f steps=%d jerr=%.3f eepos=%s"%(c,n,np.max(np.abs(qd-q)),np.round(Tach[:3,3]-np.array([b[0],b[1],0]),3)), np.round(qd,2))
env.close()
