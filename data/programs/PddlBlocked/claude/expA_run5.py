import numpy as np, fk
from expA_lib import *
from lib_util import robot, step_to
env,obs,blk = new_rig()
def trial2(env,obs,blk,a,b,c,nsub=12,pre=0.15):
    tgt=np.array([blk[0]-a,blk[1]+b,blk[2]+c]); p0=tgt-np.array([pre,0,0])
    qp,e=fk.ik(p0,fk.grasp_R(0.0),BASE,robot(obs)[3:10],seeds=8)
    if e>0.01: return ('ikfail',None,None)
    z=np.zeros(11,dtype=np.float32); z[10]=1.0; obs,_,_,_,_=env.step(z)
    obs,rjp,_=step_to(env,obs,BASE,qp)
    if rjp: return ('rej_pre',None,None)
    qc=qp.copy(); rs=None; werr=0.0
    for i in range(1,nsub+1):
        p=p0+np.array([pre*i/nsub,0,0])
        qi,ei=fk.ik(p,fk.grasp_R(0.0),BASE,qc,seeds=1); werr=max(werr,float(ei))
        obs,rj,_=step_to(env,obs,BASE,qi)
        if rj: rs=i; break
        qc=qi
    z=np.zeros(11,dtype=np.float32); z[10]=-1.0; obs,_,_,_,_=env.step(z)
    return ('ok' if rs is None else 'rej', int(robot(obs)[11]>0.5), rs)
def run(cands,tag):
    print("=="+tag)
    for (a,b,c) in cands:
        o,b0=reset(env); s=trial2(env,o,b0,a,b,c)
        print(f"a={a} b={b} c={c} -> {s}", flush=True)
run([(a,0.0,0.0) for a in [-0.06,-0.04,-0.02,-0.01,0.0,0.02,0.04,0.042,0.045,0.05,0.06,0.08,0.10]],"A")
run([(0.02,b,0.0) for b in [-0.02,-0.015,-0.012,-0.01,-0.008,0.0,0.008,0.01,0.012,0.015,0.02]],"B")
run([(0.02,0.0,c) for c in [-0.10,-0.06,-0.05,-0.04,-0.03,-0.025,-0.02,-0.01,0.0,0.02,0.04,0.06,0.07,0.075,0.08,0.10]],"C")
env.close()
