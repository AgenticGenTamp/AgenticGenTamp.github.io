"""Run seed 0 to its stable can placement, then scan box candidate slots."""
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

env=make_env(); s,info=env.reset(seed=0)
agent=GeneratedApproach(env.action_space,env.observation_space,{})
agent.reset(s,info); last=-1.0
for k in range(360):
    s,r,t,tr,_=env.step(agent.get_action(s))
    if r != last or t:
        print('EVENT',k+1,agent.phase,r,t,np.round(s[[0,1,16,17,18,32,33,34]],4).tolist(),flush=True);last=r
    if t or agent.phase>=11:break
print('POLICY_FINAL',k+1,r,t,'xyz',np.round(s[[0,1,16,17,18,32,33,34]],4).tolist(),
      'quat',np.round(s[[19,20,21,22,35,36,37,38]],4).tolist(),
      'gaps',round(agent._surface_gap(s,16,29),4),round(agent._surface_gap(s,32,45),4),flush=True)

# If unsolved, gently settle while checking whether contact release is required.
if not t:
    for j in range(15):
        s,r,t,tr,_=env.step(np.zeros(11,np.float32))
        if r != last or t:
            print('SETTLE_EVENT',j+1,r,t,np.round(s[[16,17,18,32,33,34]],4).tolist(),flush=True);last=r
        if t:break
print('FINAL',r,t,np.round(s[[0,1,16,17,18,32,33,34]],4).tolist())

# One forced extra can pass tests the fixed-slot hypothesis: Y is already at
# the tangent target, but X is 6.6 cm high, just outside the stated tolerance.
if not t:
    home=np.array([0.,-.3490659,3.1415927,-2.5481806,0.,-.8726646,1.5707964])
    pose=home.copy();pose[1]=1.0;pose[3]=-1.54
    for _ in range(45):
        e=home-s[96:103];a=np.zeros(11,np.float32);a[3:10]=np.clip(e*.5,-.1,.1)
        a[0]=np.clip((-.2365-s[93])*.5,-.1,.1);a[1]=np.clip(((s[33]-.28)-s[94])*.5,-.1,.1)
        s,r,t,tr,_=env.step(a)
        if np.max(np.abs(e))<.045 and abs(s[93]+.2365)<.025 and abs(s[94]-(s[33]-.28))<.025:break
    for _ in range(55):
        e=pose-s[96:103];a=np.zeros(11,np.float32);a[3:10]=np.clip(e*.5,-.1,.1)
        s,r,t,tr,_=env.step(a)
        if np.max(np.abs(e))<.045:break
    for j in range(45):
        a=np.zeros(11,np.float32);a[1]=.01
        s,r,t,tr,_=env.step(a)
        print('EXTRA',j+1,r,t,'can',np.round(s[32:39],4).tolist(),
              'target_err',round(float(np.hypot(s[32]-s[0],s[33]-(s[1]-(s[14]+s[46])))),4),flush=True)
        if t:break
env.close()
