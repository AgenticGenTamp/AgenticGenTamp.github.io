from calib_util import *
import json, sys
DZ=float(__import__("os").environ.get("DZ","0.012")); M0=np.array([0.12,0.0,0.0]); ZC=0.2294  # bracelet z at closed-tip table contact
def Rz(t): return np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
def grasp(env,obs,cube,yaw=0.0,off=np.zeros(2),dz=DZ):
    b,q=rstate(obs); c=objpos(obs,cube)
    a=Rz(b[2]).T@(np.array([c[0],c[1],0])-np.array([b[0],b[1],0]))-M0
    a[:2]+=off; a[2]=0.33
    Rd=down_R(yaw)
    qc,e1=ik(q,a,Rd,0); obs,_=joint_ctrl(env,obs,qc,K=1.3,steps=250,grip=0.0)
    for _ in range(8): obs=env.step(act(grip=0.0))[0]
    a[2]=ZC+dz; qc,e2=ik(qc,a,Rd,0); obs,_=joint_ctrl(env,obs,qc,K=1.3,steps=150,grip=0.0,tol=0.001)
    c_before=objpos(obs,cube)
    for _ in range(12): obs=env.step(act(grip=1.0))[0]
    a[2]=0.30; qc,_=ik(qc,a,Rd,0); obs,_=joint_ctrl(env,obs,qc,K=1.3,steps=150,grip=1.0,tol=0.001)
    for _ in range(10): obs=env.step(act(dq=np.clip(1.3*(qc-rstate(obs)[1]),-.1,.1),grip=1.0))[0]
    b,q=rstate(obs); c=objpos(obs,cube); tip,R,_,_=fk_arm(q,0)
    o=obs.get_object_from_name(cube); quat=[float(obs.get(o,f)) for f in ['qw','qx','qy','qz']]
    rec=dict(cube=cube,yaw=yaw,off=[float(x) for x in off],base=b.tolist(),q=q.tolist(),tip=tip.tolist(),R=R.tolist(),c=c.tolist(),c_before=c_before.tolist(),quat=quat,lifted=bool(c[2]>0.45))
    # release
    for _ in range(10): obs=env.step(act(grip=0.0))[0]
    return obs,rec
if __name__=='__main__':
    env=make_env(); recs=[]
    plan=json.loads(sys.argv[1]); out=sys.argv[2]
    for p in plan:  # [seed, bx,by,brot, cube, yaw, offx, offy]
        obs,_=env.reset(seed=p[0])
        obs,_=drive_base(env,obs,p[1:4],grip=0.0)
        obs,rec=grasp(env,obs,p[4],p[5],np.array(p[6:8]))
        recs.append(rec)
        b=np.array(rec['base']); t=np.array(rec['tip']); c=np.array(rec['c'])
        pred=b*[1,1,0]+Rz(b[2])@(M0+t)
        print(p, 'lifted',rec['lifted'],'c',c.round(4),'c-pred',(Rz(b[2]).T@(c-pred)).round(4), 'quat',np.round(rec['quat'],3))
    json.dump(recs,open(out,'w'))
