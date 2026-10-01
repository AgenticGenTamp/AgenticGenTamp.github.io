from env_client import make_env
import numpy as np, json
FEAT = {t['name']: t['features'] for t in json.load(open('env_spaces.json'))['observation_space']['types']}
env = make_env()
obs, info = env.reset(seed=0)
def rd(obs):
    r = obs.get_object_from_name('robot')
    return {f: round(float(obs.get(r,f)),4) for f in FEAT['Kinematic3DRobot']}
print(rd(obs))
a = np.zeros(11, dtype=np.float32)
a[3] = 0.2
for i in range(3):
    obs,rew,t,tr,info = env.step(a)
    print(rew,t,tr, rd(obs))
a=np.zeros(11,dtype=np.float32); a[0]=0.2; a[2]=0.2
for i in range(2):
    obs,rew,t,tr,info = env.step(a)
    print(rd(obs))
# huge joint move to test limits
a=np.zeros(11,dtype=np.float32); a[4]=0.2
for i in range(15):
    obs,rew,t,tr,info = env.step(a)
print(rd(obs))
