"""Grid-search a collision-safe green side grasp after southeast blocker removal."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np
import sys

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 120
for dx in np.arange(.025, .176, .025):
  for dy in np.arange(-.175, -.024, .025):
    env=make_env(); state,info=env.reset(seed=seed)
    p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(state,info)
    for _ in range(80):
      state,_,_,_,_=env.step(p.get_action(state))
      if p.stage==5:break
    # Set elbow extension while clear of the pen.
    for _ in range(2):
      a=np.zeros(11,np.float32);a[6]=np.clip(p.Q[3]+.08-p.g(state,'robot','joint_4'),-.2,.2);a[10]=1
      state,_,_,_,_=env.step(a)
    target=p.target(p.green)+np.array([dx,dy])
    # Axis-aligned return prevents diagonal wall clipping.
    corner=np.array([target[0],p.robot(state)[1]])
    for waypoint,lift in ((corner,True),(target,True),(target,False)):
      for _ in range(15):
        if p.at(state,waypoint,arm=not lift):break
        state,_,_,_,_=env.step(p.motion(state,waypoint,1,lift=lift,q4add=.08))
    a=np.zeros(11,np.float32);a[10]=-1;state,_,_,_,_=env.step(a)
    held=p.g(state,'robot','grasp_active')>.5
    env.close()
    if held:
      print('HIT',seed,round(dx,3),round(dy,3),'out',np.round(p.out,3));sys.exit(0)
print('NO HIT',seed)
