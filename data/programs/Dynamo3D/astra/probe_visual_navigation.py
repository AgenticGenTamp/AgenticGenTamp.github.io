import numpy as np
from env_client import make_env
for target in [(2,0),(3,0),(3,3),(2,4),(4,2),(4,4)]:
    env=make_env()
    s,info=env.reset(seed=1)
    robot=next(iter(s.get_objects(env.observation_space.get_type('mujoco_tidybot_robot'))))
    total=0
    for step in range(180):
        p=np.array([s.get(robot,'pos_base_x'),s.get(robot,'pos_base_y')])
        a=np.zeros(11,dtype=np.float32)
        a[:2]=np.clip(np.array(target)-p,-.1,.1)
        s,r,done,trunc,info=env.step(a)
        total+=r
        if done or trunc or np.linalg.norm(np.array(target)-p)<.04:
            break
    p=[s.get(robot,'pos_base_x'),s.get(robot,'pos_base_y')]
    print(target,'steps',step+1,'pos',p,'reward',total,'last',r,'done',done,'trunc',trunc,'info',info,flush=True)
    env.close()
