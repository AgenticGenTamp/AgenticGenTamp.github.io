import numpy as np, json, fk
from expA_lib import *
from lib_util import robot, step_to
env,obs,blk = new_rig()
# diagnose c=0.01 : compare IK solutions along path
o,b0 = reset(env)
tgt=np.array([b0[0]-0.03,b0[1],b0[2]+0.01]); p0=tgt-np.array([0.15,0,0])
qp,_=fk.ik(p0,fk.grasp_R(0.0),BASE,robot(o)[3:10],seeds=8)
qc=qp.copy()
for i in range(1,7):
    p=p0+np.array([0.15*i/6,0,0])
    qi,ei=fk.ik(p,fk.grasp_R(0.0),BASE,qc,seeds=2)
    print(i, round(float(np.linalg.norm(qi-qc)),3), round(float(ei),4))
    qc=qi
# retry c=0.01 with nsub=12 and seeds=1 warm start (smoother)
def trial2(env,obs,blk,a,b,c,nsub=12):
    tgt=np.array([blk[0]-a,blk[1]+b,blk[2]+c]); p0=tgt-np.array([0.15,0,0])
    q=robot(obs)[3:10]
    qp,e=fk.ik(p0,fk.grasp_R(0.0),BASE,q,seeds=8)
    z=np.zeros(11,dtype=np.float32); z[10]=1.0; obs,_,_,_,_=env.step(z)
    obs,rj,_=step_to(env,obs,BASE,qp)
    qc=qp.copy(); rs=None
    for i in range(1,nsub+1):
        p=p0+np.array([0.15*i/nsub,0,0])
        qi,ei=fk.ik(p,fk.grasp_R(0.0),BASE,qc,seeds=1)
        obs,rj,_=step_to(env,obs,BASE,qi)
        if rj: rs=i; break
        qc=qi
    z=np.zeros(11,dtype=np.float32); z[10]=-1.0; obs,_,_,_,_=env.step(z)
    return int(robot(obs)[11]>0.5), rs
for c in [0.01,0.0,0.02]:
    o,b0=reset(env); print("nsub12 c=",c, trial2(env,o,b0,0.03,0.0,c), flush=True)
# robustness repeats of recommended point
for k in range(3):
    o,b0=reset(env); o,r=trial(env,o,b0,0.02,0.0,0.03); print("rec",r.get('ga'),r['status'])
env.close()
