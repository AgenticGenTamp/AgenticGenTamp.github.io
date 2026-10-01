from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = env.observation_space.get_type('mujoco_tidybot_robot')
r = obs.get_objects(R)[0]
J = ['pos_arm_joint%d'%i for i in range(1,8)]
q = lambda o: np.array([o.get(r,k) for k in J])
q0 = q(obs); tgt = q0.copy(); tgt[0]+=0.5; tgt[1]+=0.4; tgt[3]+=0.3
for t in range(30):
    a = np.zeros(11); a[3:10] = np.clip(tgt - q(obs), -0.1, 0.1)
    obs, rew, te, tr, inf = env.step(a)
    if t%3==0: print(t, np.round(q(obs)-tgt,3), rew)
env.close()
