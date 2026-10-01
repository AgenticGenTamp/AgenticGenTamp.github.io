import numpy as np
from env_client import make_env
from probe_fk_world import arm_from_world, RDOWN_ARM, ee_world
from ik import jacobian
np.set_printoptions(precision=4,suppress=True,linewidth=250)
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
def sv(o,p_w,steps,k,grip=0.0,log=20,integ=0.0):
    acc=np.zeros(7)
    for i in range(steps):
        q=o[128:135].copy(); pa=arm_from_world(o[125:128],p_w)
        J,M=jacobian(q); ep=pa-M[:3,3]
        Rerr=RDOWN_ARM@M[:3,:3].T
        ang=np.arccos(np.clip((np.trace(Rerr)-1)/2,-1,1))
        ax=np.zeros(3)
        if ang>1e-8:
            ax=np.array([Rerr[2,1]-Rerr[1,2],Rerr[0,2]-Rerr[2,0],Rerr[1,0]-Rerr[0,1]])/(2*np.sin(ang))*ang
        e=np.concatenate([ep,ax])
        dq=J.T@np.linalg.solve(J@J.T+0.08**2*np.eye(6),e)
        acc=acc*0.98+dq*integ
        a=np.zeros(11); a[3:10]=np.clip(dq*k+acc,-0.1,0.1); a[10]=grip
        o,_,_,_,_=env.step(a); o=np.asarray(o,float)
        if i%log==0 or i==steps-1:
            print(f"  i={i:3d} |ep|={np.linalg.norm(ep):.4f} ang={ang:.3f} maxcmd={np.abs(a[3:10]).max():.4f} ee={np.round(ee_world(o),4)}")
    return o
print("== target A (converged before)"); o=sv(o,[0.90,-0.20,0.55],60,2.5)
print("== target B (stalled before), k=2.5"); o=sv(o,[0.75,-0.15,0.62],120,2.5)
print("== same, k=8"); o=sv(o,[0.75,-0.15,0.62],80,8.0)
print("== same, k=2.5 + integral"); o=sv(o,[0.75,-0.15,0.62],80,2.5,integ=0.6)
print("== hold zero cmd 30 steps (drift under gravity)")
for i in range(30):
    a=np.zeros(11); a[10]=0.0; o,_,_,_,_=env.step(a); o=np.asarray(o,float)
print("  ee after hold",np.round(ee_world(o),4))
print("== deadband test: constant small joint cmds on j1")
for c in [0.002,0.005,0.01,0.02,0.05]:
    q0=o[128:135].copy()
    for i in range(10):
        a=np.zeros(11); a[3]=c; a[10]=0.0; o,_,_,_,_=env.step(a); o=np.asarray(o,float)
    print(f"  cmd={c}: dq1 over 10 steps = {o[128]-q0[0]:+.5f} (ratio {(o[128]-q0[0])/10/c:.3f})")
env.close()
