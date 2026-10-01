from kin_util import *
obs,_=env.reset(seed=1)
print(blocks(obs))
obs,ok=move_to(obs,3.0,1.5); print(ok,R(obs))
for th in [math.pi/2, math.pi/2+0.3, math.pi/2+0.8]:
  for a in [0.2,0.5,0.8]:
    obs,ok=move_to(obs,3.0,1.2); obs=set_theta(obs,th); obs=set_arm(obs,a)
    obs=push_limit(obs,0,0.05); r=R(obs)
    print('th%.3f a%.1f ymax %.4f  tip_y %.4f'%(r[2],a,r[1],r[1]+a*math.sin(r[2])))
for th in [0.0,0.3]:
  for a in [0.2,0.8]:
    obs,ok=move_to(obs,3.5,1.2); obs=set_theta(obs,th); obs=set_arm(obs,a)
    obs=push_limit(obs,0.05,0); r=R(obs)
    print('th%.3f a%.1f xmax %.4f  tip_x %.4f'%(r[2],a,r[0],r[0]+a*math.cos(r[2])))
print(blocks(obs))
