from plib import *
A=0.15
lo,hi=0.08,0.09
for i in range(8):
    off=(lo+hi)/2
    env=make_env(); obs,_=env.reset(seed=0)
    tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
    obs,_=setarm(env,obs,A); obs,_=setx(env,obs,cx+off); obs,_=sety(env,obs,top+0.005+A+0.005)
    obs,*_=step(env,v=1.0); y0=d(obs,'target_block')['y']
    obs,*_=step(env,dy=0.05,v=1.0)
    if d(obs,'target_block')['y']-y0>0.04: lo=off
    else: hi=off
    env.close()
print(f"x-offset threshold (gap 0.005): works<={lo:.5f} fails>={hi:.5f}; block half-width=0.0502 -> excess={lo-0.0502:.5f}")
# same with gap 0.0 to see if threshold depends on vertical gap
lo,hi=0.08,0.10
for i in range(8):
    off=(lo+hi)/2
    env=make_env(); obs,_=env.reset(seed=0)
    tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
    obs,_=setarm(env,obs,A); obs,_=setx(env,obs,cx+off); obs,_=sety(env,obs,top+A+0.005)
    obs,*_=step(env,v=1.0); y0=d(obs,'target_block')['y']
    obs,*_=step(env,dy=0.05,v=1.0)
    if d(obs,'target_block')['y']-y0>0.04: lo=off
    else: hi=off
    env.close()
print(f"x-offset threshold (gap 0.000): works<={lo:.5f} fails>={hi:.5f}; excess={lo-0.0502:.5f}")
