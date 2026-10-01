from env_client import make_env
import numpy as np, json
sp = json.load(open("env_spaces.json"))["observation_space"]["types"]
TF = {t["name"]: t["features"] for t in sp}
def rob(o):
    r=o.get_object_from_name("robot")
    return {f: round(float(o.get(r,f)),4) for f in TF["Kinematic3DRobot"]}
env = make_env()
obs,info = env.reset(seed=0)
print("start", rob(obs))
a=np.zeros(11); a[3]=0.4
obs,r,t,tr,i = env.step(a)
print("j1+0.4", rob(obs), r,t,tr)
a=np.zeros(11); a[0]=0.4; a[1]=-0.4; a[2]=0.3
obs,r,t,tr,i = env.step(a)
print("base", rob(obs))
a=np.zeros(11); a[10]=-1.0
obs,r,t,tr,i = env.step(a)
print("close", rob(obs))
a=np.zeros(11); a[10]=1.0
obs,r,t,tr,i = env.step(a)
print("open", rob(obs))
# large joint move
for k in range(10):
    a=np.zeros(11); a[4]=0.4
    obs,r,t,tr,i=env.step(a)
print("j2 x10", rob(obs))
env.close()
