from plib import *
env=make_env(); obs,_=env.reset(seed=0)
print("arm start",d(obs,'robot')['arm_joint'])
for i in range(5):
    obs,*_=step(env,da=-0.1); print("  retract ->",round(d(obs,'robot')['arm_joint'],6))
for i in range(4):
    obs,*_=step(env,da=0.1); print("  extend ->",round(d(obs,'robot')['arm_joint'],6))
env.close()
