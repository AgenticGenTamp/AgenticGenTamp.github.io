from plib import *
A=0.15
print("== TASK4b: grasp target_block, release, then grasp a DIFFERENT object (seed 13)")
env=make_env(); obs,_=env.reset(seed=13)
tb=d(obs,'target_block'); print("  tb",{k:round(v,4) for k,v in tb.items() if k in('x','y','width','height')})
cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
obs,_=setarm(env,obs,A); obs,_=setx(env,obs,cx); obs,_=sety(env,obs,top+0.005+A+0.005)
obs,*_=step(env,v=1.0)
for i in range(4): obs,*_=step(env,dy=0.05,v=1.0)
print("  lifted tb y",round(d(obs,'target_block')['y'],4))
obs,*_=step(env,v=0.0); obs,*_=step(env,v=0.0)
print("  released tb y",round(d(obs,'target_block')['y'],4))
o0=d(obs,'obstruction0'); c2=o0['x']+o0['width']/2; t2=o0['y']+o0['height']
obs,_=sety(env,obs,0.7,vac=0); obs,_=setx(env,obs,c2,vac=0)
obs,_=sety(env,obs,t2+0.005+A+0.005,vac=0)
obs,*_=step(env,v=1.0); y0=d(obs,'obstruction0')['y']
obs,*_=step(env,dy=0.05,v=1.0)
print("  DIFFERENT-object grasp dy:",round(d(obs,'obstruction0')['y']-y0,5))
env.close()
print("== X-offset tolerance (block width 0.1004, gap 0.005), seed 0")
for off in [0.05,0.055,0.06,0.07,0.08,0.09]:
    env=make_env(); obs,_=env.reset(seed=0)
    tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
    obs,_=setarm(env,obs,A); obs,_=setx(env,obs,cx+off); obs,ok=sety(env,obs,top+0.005+A+0.005)
    obs,*_=step(env,v=1.0); y0=d(obs,'target_block')['y']
    obs,*_=step(env,dy=0.05,v=1.0)
    print(f"  offset={off:.3f} (block half-width 0.0502) yok={ok} grasp={'YES' if d(obs,'target_block')['y']-y0>0.04 else 'NO'}")
    env.close()
