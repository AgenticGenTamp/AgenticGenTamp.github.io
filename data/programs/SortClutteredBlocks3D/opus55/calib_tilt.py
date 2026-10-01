from calib_util import *
import sys, json
def rot(ax,t):
    ax=np.array(ax,float); K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*K@K
env=make_env(); res=[]
grip=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
for axname,ax in [('y',[0,1,0]),('x',[1,0,0])]:
  for deg in [0,15,30,45,-15,-30,-45]:
    if axname=='x' and deg==0: continue
    obs,_=env.reset(seed=0)
    for _ in range(3): obs=env.step(act(grip=grip))[0]
    obs,_=drive_base(env,obs,[0.62,0.0,np.pi],grip=grip)
    b,q=rstate(obs)
    Rd=rot(ax,np.radians(deg))@down_R(0)
    tgt=np.array([0.36,0.0,0.30])
    qc,_=ik(q,tgt,Rd,0); obs,_=joint_ctrl(env,obs,qc,K=1.3,steps=200,grip=grip)
    zc=None
    for i in range(300):
        tgt=tgt+np.array([0,0,-0.004 if tgt[2]>0.27 else -0.001])
        qc,e=ik(qc,tgt,Rd,0)
        obs,_=joint_ctrl(env,obs,qc,K=1.3,steps=4,grip=grip,tol=0.0)
        b,q=rstate(obs); tip=fk_arm(q,0)[0]
        if tip[2]-tgt[2]>0.0015: zc=tip[2]; break
    print(axname,deg,'contact bracelet z',None if zc is None else round(zc,4),'ikerr',round(e,5))
    res.append((axname,deg,zc))
json.dump(res,open(f'calib_tilt_{grip}.json','w'))
