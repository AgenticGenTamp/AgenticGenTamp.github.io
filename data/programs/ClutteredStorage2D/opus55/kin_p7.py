from kin_util import *
obs,_=env.reset(seed=0); b0=blocks(obs)
obs,_=move_to(obs,R(obs)[0],0.3); obs,_=move_to(obs,4.0,0.3)
obs=set_theta(obs,math.pi); obs=set_arm(obs,0.8); print(R(obs))
obs=push_limit(obs,0,0.05); r=R(obs); print('arm-link sweep up: stopped at y %.4f'%r[1], r)
print('blocks unchanged', blocks(obs)==b0)
# rotation collision: partial?
obs,_=move_to(obs,4.0,0.3); obs=set_theta(obs,math.pi); r0=R(obs)
obs2,*_=env.step(A(dt=-0.196)); print('rot', r0, R(obs2))
