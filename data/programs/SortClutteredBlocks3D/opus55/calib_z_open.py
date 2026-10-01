from calib_util import *
import sys
env=make_env(); obs,_=env.reset(seed=0)
obs=env.step(act(grip=0.0))[0]
obs,k=drive_base(env,obs,[0.55,0.0,np.pi],grip=0.0); b,q=rstate(obs); print('base',b.round(4),k)
Rd=down_R(0)
tgt=np.array([0.35,0.0,0.45])  # arm frame guess
qc=q.copy()
q_cmd,err=ik(q,tgt,Rd,tool=0); print('ik err',err, q_cmd.round(3))
obs,k=joint_ctrl(env,obs,q_cmd,K=1.3,steps=200,grip=0.0); print('reach',k)
for i in range(200):
    tgt=tgt+np.array([0,0,-0.004])
    q_cmd,err=ik(q_cmd,tgt,Rd,tool=0)
    obs,k=joint_ctrl(env,obs,q_cmd,K=1.3,steps=6,grip=0.0)
    b,q=rstate(obs); tip,R,_,_=fk_arm(q,0)
    e=np.linalg.norm(tip-tgt)
    if i%5==0 or e>0.005: print(i,'tgt z',tgt[2].round(4),'tip',tip.round(4),'err',round(e,4), 'jerr',np.abs(q_cmd-q).max().round(4))
    if e>0.02: break
