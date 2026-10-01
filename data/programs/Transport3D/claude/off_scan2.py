import sys, numpy as np, kutil, kin, off_lib2 as L
from env_client import make_env
np.set_printoptions(precision=4,suppress=True)
seed=int(sys.argv[1]); yaw_off=float(sys.argv[2]); tag=sys.argv[3]; TZ=0.024
env=make_env(); o,info=env.reset(seed=seed)
c=kutil.Ctl(env,o); cu0=c.opos("cube1")
c.gotobase(cu0[0]+0.42,cu0[1],np.pi)
b,q,g=c.robot(); yaw=b[2]+yaw_off
HI=0.16
recs=[]
def goover(x,y):
    T=L.fkq(c)
    if L.vpath(c,[T[0,3],T[1,3],HI],yaw,step=0.04) and L.vpath(c,[x,y,HI],yaw,step=0.04): return True
    ok,e=L.vmove(c,[x,y,HI],yaw)
    if ok: return True
    return L.vpath(c,[x,y,HI],yaw,step=0.05)
if not goover(cu0[0],cu0[1]):
    print("### %s PREGRASP_FAIL"%tag); env.close(); sys.exit()
def trial(dx,dy,dz):
    cu=c.opos("cube1")
    tp=np.array([cu[0]+dx,cu[1]+dy,cu[2]+TZ+dz])
    if not goover(tp[0],tp[1]): return None
    if not L.vpath(c,tp,yaw,step=0.03): return None
    T=L.fkq(c); b2,q2,_=c.robot()
    off_t=T[:3,:3].T@(cu-T[:3,3])     # cube centre in tool frame
    c.grip(True); s=1 if c.robot()[2]>0.5 else 0
    if s:
        c.grip(False)
        if c.robot()[2]>0.5: return None
    recs.append((dx,dy,dz,s,off_t,q2.copy()))
    return s
G=[-0.03,-0.015,0.0,0.015,0.03]
grid={}
for dy in G:
    for dx in G:
        r=trial(dx,dy,0.0); grid[(dx,dy)]='X' if r==1 else ('.' if r==0 else 'm')
        if c.steps>860: break
    if c.steps>860: break
zres=[]
for dz in [-0.02,-0.01,0.0,0.01,0.02,0.03]:
    if c.steps>950: break
    r=trial(0,0,dz); zres.append((dz,'X' if r==1 else ('.' if r==0 else 'm')))
print("### %s yaw_off=%.2f yaw=%.3f steps=%d"%(tag,yaw_off,yaw,c.steps))
print("  dx:"+" ".join("%6.3f"%d for d in G))
for dy in G: print("dy%6.3f "%dy+"     ".join(grid.get((dx,dy),'?') for dx in G))
print("zscan(dx=dy=0):"," ".join("%+.2f:%s"%z for z in zres))
S=[r for r in recs if r[3]==1]; F=[r for r in recs if r[3]==0]
for nm,A in [("SUCC",S),("FAIL",F)]:
    if A:
        M=np.array([r[4] for r in A])
        print("%s n=%d off_tool min=%s max=%s mean=%s"%(nm,len(A),np.round(M.min(0),4),np.round(M.max(0),4),np.round(M.mean(0),4)))
if S:
    Q=np.array([r[5] for r in S]); print("j5,j6,j7 range",np.round(Q[:,4:].min(0),2),np.round(Q[:,4:].max(0),2))
np.save("off2_%s.npy"%tag,np.array([np.concatenate([[r[0],r[1],r[2],r[3]],r[4],r[5]]) for r in recs]))
env.close()
