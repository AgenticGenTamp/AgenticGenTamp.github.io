from common import *
env=make_env(); s,_=env.reset(seed=0,options={'object_count':4})
print(rob(s))
for a in [[0.03,0,0,0,0]]*3+[[0,-0.03,0.05,0.08,-0.015]]*3+[[0,0,0,0,0]]:
    s,*_=env.step(np.array(a)); print({k:round(v,4) for k,v in rob(s).items()})
