from kin_util import *
for seed in [1,0]:
  obs,_=env.reset(seed=seed)
  s=obs.get_object_from_name('shelf')
  x1,w1=obs.get(s,'x1'),obs.get(s,'width1'); print('seed',seed,'x1',x1,'w1',w1)
  print(blocks(obs))
  r=R(obs); obs,ok=move_to(obs,r[0],2.2); print(ok,R(obs))
  cx=x1+w1/2 if seed==1 else x1+w1-0.1
  obs,ok=move_to(obs,cx,2.2); print(ok,R(obs))
  obs=set_theta(obs,math.pi/2); obs=set_arm(obs,0.2)
  obs=push_limit(obs,0,0.05); r=R(obs); print('base ymax',r)
  seq=[]
  for k in range(7):
    o2,*_=env.step(A(da=0.1)); seq.append(R(o2)[3]); obs=o2
  print('extend',seq)
  r=R(obs); print('tip y', r[1]+r[3], 'grip top',r[1]+r[3]+0.01)
  o=push_limit(obs,-0.05,0); r=R(o); print('left lim x',r[0], 'grip left edge', r[0]-0.07)
  o=push_limit(o,0.05,0); r=R(o); print('right lim x',r[0],'grip right edge', r[0]+0.07)
  print(blocks(o))
