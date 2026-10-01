from grasp_utils import *
env = new_env()
obs, info = env.reset(seed=0)
r = rob(obs); print('robot', r)
# test frame of dx
obs,*_ = env.step(act(dx=0.05)); print('dx', rob(obs)-r); r=rob(obs)
obs,*_ = env.step(act(dy=0.05)); print('dy', rob(obs)-r); r=rob(obs)
obs,*_ = env.step(act(dth=0.1)); print('dth', rob(obs)-r); r=rob(obs)
obs,*_ = env.step(act(darm=0.1)); print('darm', rob(obs)-r); r=rob(obs)
obs,*_ = env.step(act(darm=0.1)); print('darm', rob(obs)-r); r=rob(obs)
for i in range(6):
  obs,*_ = env.step(act(darm=0.1)); 
print('arm after', rob(obs))
obs,*_ = env.step(act(vac=0.4)); print('vac.4', rob(obs))
obs,*_ = env.step(act(vac=0.6)); print('vac.6', rob(obs))
obs,*_ = env.step(act(darm=-0.1)); print('darm-', rob(obs))
