"""After current seed0 policy, gently reduce box-bowl gap then fully retreat."""
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

env=make_env(); s,info=env.reset(seed=0); home=s[96:103].copy()
agent=GeneratedApproach(env.action_space,env.observation_space,{})
agent.reset(s,info)
for k in range(500):
 s,r,t,tr,_=env.step(agent.get_action(s))
 if t or tr or agent.phase>=11: break
print('baseline',k,r,t,np.round(s[[0,1,16,17,18,32,33,34]],4).tolist(),flush=True)

def do(a):
 global s
 s,r,t,tr,_=env.step(np.asarray(a,np.float32)); return r,t

# Re-align outside box and deploy open broad pusher.
for _ in range(45):
 a=np.zeros(11); gy=float(s[17])+.28; gx=float(s[16])-.542
 a[1]=np.clip((gy-s[94])*.5,-.1,.1); a[0]=np.clip((gx-s[93])*.5,-.1,.1)
 r,t=do(a)
 if abs(gy-s[94])<.025 and abs(gx-s[93])<.02:break
pose=home.copy();pose[1]=1.;pose[3]=-1.70
for _ in range(75):
 e=pose-s[96:103]
 if np.max(np.abs(e))<.04:break
 a=np.zeros(11);a[3:10]=np.clip(e*.5,-.1,.1);r,t=do(a)
for j in range(4):
 before=s.copy();a=np.zeros(11);a[1]=-.1;r,t=do(a)
 print('nudge',j,r,t,'box',np.round(s[16:19],4).tolist(),'bowl',np.round(s[:3],4).tolist(),flush=True)
 if t:break
# Radial retreat and arm fold, with reward checks.
for j in range(40):
 a=np.zeros(11);a[0]=np.clip(((float(s[93])-.30)-s[93])*.5,-.1,.1) if j==0 else -.1
 r,t=do(a)
 if t:
  print('TERMINAL retreat',j,r,'objects',np.round(s[[0,1,2,16,17,18,32,33,34]],6).tolist(),
        'quats',np.round(s[[3,4,5,6,19,20,21,22,35,36,37,38]],6).tolist(),
        'basearm',np.round(s[93:104],5).tolist(),flush=True)
  env.close();raise SystemExit
for j in range(75):
 e=home-s[96:103];a=np.zeros(11);a[3:10]=np.clip(e*.5,-.1,.1);r,t=do(a)
 if t:print('fold success',j,r);break
 if np.max(abs(e))<.04:break
for j in range(15):
 r,t=do(np.zeros(11));
 if t or r!=-1:print('settle event',j,r,t);break
print('final',r,t,np.round(s[[0,1,16,17,18,32,33,34]],4).tolist(),flush=True)
env.close()
