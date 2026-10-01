from kin_util import *
def mt(obs,x,y):
    # move y-first safe path: go down to 2.2 then x then y
    r=R(obs); obs,_=move_to(obs,r[0],min(r[1],2.3)); obs,_=move_to(obs,x,R(obs)[1]); obs,_=move_to(obs,x,y); return obs
for seed in [2,3]:
  obs,_=env.reset(seed=seed)
  s=obs.get_object_from_name('shelf')
  x1,w1=obs.get(s,'x1'),obs.get(s,'width1'); print('seed',seed,'purple x',round(x1,4),round(x1+w1,4), blocks(obs))
  r=R(obs); obs,ok=move_to(obs,r[0],2.2)
  obs=mt(obs,x1+w1/2,2.2); obs=set_theta(obs,math.pi/2)
  for gt in [2.626,2.63,2.64,2.645,2.65,2.655,2.66,2.665]:
    obs=mt(obs,x1+w1/2,gt-0.21)
    if abs(R(obs)[1]-(gt-0.21))>1e-3: print('cannot reach',R(obs)); continue
    obs=push_limit(obs,-0.05,0); L=R(obs)[0]-0.07
    obs=push_limit(obs,0.05,0); Rr=R(obs)[0]+0.07
    print(' gt',gt,'L grip edge %.4f  R grip edge %.4f'%(L,Rr))
