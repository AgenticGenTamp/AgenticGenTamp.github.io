import numpy as np
from env_client import make_env
from approach import GeneratedApproach, Robot, tool_R, yaw_of
env=make_env(); obs,info=env.reset(seed=167)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
R=Robot(obs); b=R.blocks["block0"]
byaw=yaw_of(*b[3:7])
found=[]
for rot in [0.0,-0.2,-0.4]:
  for by in np.arange(-0.22,0.23,0.055):
    for tilt in [0.3,0.45,0.6,0.75,0.9]:
        for td in np.arange(-np.pi,np.pi,np.pi/6):
            for ang in [byaw, byaw+np.pi/2]:
                Rw=tool_R(ang,tilt,td)
                gp=np.array(b[:3])-0.05*Rw[:,0]
                pre=gp-0.13*Rw[:,0]
                base=(-0.43,by,rot)
                if ap.ik_at(base,gp,Rw) is not None and ap.ik_at(base,pre,Rw) is not None:
                    found.append((round(rot,2),round(by,3),tilt,round(td,2)))
print(len(found))
import collections
print(collections.Counter([(f[0],f[2]) for f in found]))
print(found[:10])
env.close()
