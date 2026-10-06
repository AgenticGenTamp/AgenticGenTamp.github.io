import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:Dynamo3DEnv",
    count_kwarg="num_objects", count_object_prefix="obstacle_chair",
    constant_object_env_kwargs={'scene_bg': False}, design_counts=[1,3], eval_counts=[1,3])
for cnt in [1,3]:
    for s in range(6):
        obs,_ = env.reset(seed=s, options={"object_count": cnt})
        r=[o for o in obs if o.name=="robot"][0]
        ch=[(o.name, np.round(obs[o][:3],2).tolist(), round(float(2*np.arctan2(obs[o][6],obs[o][3])),2)) for o in obs if o.name!="robot"]
        print(cnt, s, "robot", np.round(obs[r][:3],3).tolist(), ch)
obs,_ = env.reset(seed=0, options={"object_count": 3}); r=[o for o in obs if o.name=="robot"][0]
for name,a in [("dx+0.1",[0.1,0,0]),("dy+0.1",[0,0.1,0]),("dth+0.1",[0,0,0.1])]:
    obs,_ = env.reset(seed=0, options={"object_count": 3}); p0=obs[r][:3].copy()
    for i in range(12):
        act=np.zeros(11); act[:3]=a
        obs,rew,te,tr,_=env.step(act)
        if i in (0,1,2,5,11): print(name, i, np.round(obs[r][:3]-p0,4).tolist(), "vel", np.round(obs[r][11:14],3).tolist())
    for i in range(10): obs,*_=env.step(np.zeros(11))
    print(name, "after 10 zero steps", np.round(obs[r][:3]-p0,4).tolist())
