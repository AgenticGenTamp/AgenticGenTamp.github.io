from kin_util import *
obs,_=env.reset(seed=0); b0=blocks(obs); print(b0['block3'],b0['block4'])
obs,_=move_to(obs,R(obs)[0],0.3); obs,_=move_to(obs,4.05,0.3)
obs=set_theta(obs,math.pi/2); obs=set_arm(obs,0.8)
obs=push_limit(obs,0,0.05); r=R(obs); print('from below x=4.05: grip top y %.4f'%(r[1]+0.81), r)
obs,_=move_to(obs,4.05,0.3); obs,_=move_to(obs,4.4,0.3); obs,_=move_to(obs,4.4,1.62); print(R(obs))
obs=set_theta(obs,math.pi); obs=set_arm(obs,0.3); print(R(obs))
obs=push_limit(obs,-0.05,0); r=R(obs); print('from right y=1.62: grip left x %.4f'%(r[0]-0.31), r)
print('blocks unchanged', blocks(obs)==b0)
