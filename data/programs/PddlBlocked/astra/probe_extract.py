from env_client import make_env
from approach import GeneratedApproach
from kinematics import fk,solve
import numpy as np

fs=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
for mode in ['raisedcp']:
 e=make_env();s,info=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{})
 p.reset(s,info);r=s.get_object_from_name('robot')
 for t in range(200):
  a=p.get_action(s);s,*_=e.step(a)
  if s.get(r,'grasp_active')>.5:break
 else: print('FAILED_GRASP',mode);e.close();continue
 cfg=np.array([s.get(r,f) for f in fs]);old=cfg.copy();tool,R=fk(cfg)
 if mode=='raisedcp':
  a=np.zeros(11);a[4]=-.03;s,*_=e.step(a);old=np.array([s.get(r,f) for f in fs]);cfg=old.copy();cfg[:2]=p.cp[:2];tool,_=fk(old)
 elif mode in ['cp','cp_then_high']:cfg=p.cp.copy()
 elif mode=='cp_exactq':cfg[:2]=p.cp[:2]
 elif mode=='high':cfg=p.high.copy()
 elif mode=='retreat':cfg[:2]-=.1*p.d[:2]
 elif mode=='baseleft':cfg[0]-=.1
 elif mode=='basenorth':cfg[1]+=.1
 elif mode.startswith('q2up'):cfg[4]-=int(mode[-2:])*.01
 else:
  dz=int(mode[-2:])*.01;goal=tool+np.array([0,0,dz])
  if mode.startswith('diag'):goal-=.05*p.d
  cfg,err=solve(goal,p.yaw,cfg[:2],cfg)
  print('IK',mode,err,flush=True)
 a=np.zeros(11);d=cfg-old;d[[2,7,9]]=(d[[2,7,9]]+np.pi)%(2*np.pi)-np.pi;a[:10]=d*min(1.,.2/max(abs(d)))
 s,*_=e.step(a);new=np.array([s.get(r,f) for f in fs]);newtool,_=fk(new)
 print(mode,'grasp_at',t,'moved',max(abs(new-old)),'tool_delta',(newtool-tool).round(5).tolist(),'red',[s.get(s.get_object_from_name('blocker'),'pose_'+ax) for ax in 'xyz'],flush=True)
 if mode=='cp_then_high':
  old=new.copy();d=p.high-old;d[[2,7,9]]=(d[[2,7,9]]+np.pi)%(2*np.pi)-np.pi;a=np.zeros(11);a[:10]=d*min(1.,.2/max(abs(d)));s,*_=e.step(a);new=np.array([s.get(r,f) for f in fs]);print('THEN_HIGH',max(abs(new-old)),flush=True)
 e.close()
