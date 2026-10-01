from cal_util import *
env=make_env(); obs,_=env.reset(seed=0)
s=env.get_state(); print(type(s))
R=s.get_object_from_name('robot'); s.set(R,'joint_1',0.5)
env.set_state(s)
o,*_=env.step(act()); print(rob(o)['q'])
# open gripper
o,*_=env.step(act(grip=1.0)); print(rob(o)['finger'])
o,*_=env.step(act(grip=-1.0)); r=rob(o); print(r['finger'],r['ga'])
