from calib_util import *
from calib_touch import moveto, SAFE, ZC, Rz, M0
from calib_tilt_util import rot
import json, sys
env=make_env(); recs=[]
plan=json.loads(sys.argv[1])  # list of [seed, bx,by,brot, cube]
for seed,bx,by,br,cube in plan:
    obs,_=env.reset(seed=seed)
    obs,_=drive_base(env,obs,[bx,by,br])
    b,_=rstate(obs); yawg=-wrap(b[2]-np.pi)
    c=objpos(obs,cube); p=c.copy(); p[2]=SAFE
    obs,_=moveto(env,obs,p,yawg,grip=0.0)
    p[2]=ZC-0.006; obs,_=moveto(env,obs,p,yawg,grip=0.0,tol=0.001)
    for _ in range(12): obs=env.step(act(grip=1.0))[0]
    b,q=rstate(obs); a=Rz(b[2]).T@(np.array([c[0],c[1],0])-np.array([b[0],b[1],0]))-M0; a[2]=0.40
    for ax,th,yw in [([1,0,0],0,0),([1,0,0],0.4,0),([1,0,0],-0.4,0),([0,1,0],0.4,0),([0,1,0],-0.4,0),([1,0,0],0,0.8),([1,1,0],0.35,-0.8),([1,-1,0],0.35,0.3),([0,1,0],0.3,1.2),([1,0,0],0,0)]:
        Rd=rot(np.array(ax)/np.linalg.norm(ax),th)@down_R(yawg+yw)
        b,q=rstate(obs); qc,e=ik(q,a,Rd,0)
        obs,_=joint_ctrl(env,obs,qc,K=1.3,steps=200,grip=1.0,tol=0.001)
        for _ in range(10): obs=env.step(act(dq=np.clip(1.3*(qc-rstate(obs)[1]),-.1,.1),grip=1.0))[0]
        b,q=rstate(obs); o=obs.get_object_from_name(cube)
        cq=[float(obs.get(o,f)) for f in ['qw','qx','qy','qz']]
        c=objpos(obs,cube)
        recs.append(dict(seed=seed,base=b.tolist(),q=q.tolist(),c=c.tolist(),cq=cq,ik=float(e)))
        t,R,_,_=fk_arm(q,0)
        print(seed,ax,th,yw,'c',c.round(4),'ik',round(e,5), 'held',c[2]>0.45)
json.dump(recs,open(sys.argv[2],'w'))
