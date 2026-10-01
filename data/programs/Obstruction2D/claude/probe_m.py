from plib import *
A=0.15
env=make_env(); obs,_=env.reset(seed=0)
tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
obs,_=setarm(env,obs,A); obs,_=setx(env,obs,cx); obs,_=sety(env,obs,top+0.005+A+0.005)
obs,*_=step(env,v=1.0)
obs,_=sety(env,obs,0.35,vac=1)
# push block right into obstruction1 (left face 0.68) at block y=0.1 level
delta=0.05
for i in range(30):
    p=d(obs,'robot')['x']; obs,*_=step(env,dx=delta,v=1.0)
    if abs(d(obs,'robot')['x']-p)<1e-9:
        delta/=2
        if delta<1e-7: break
r0=d(obs,'robot'); print("at contact robot",round(r0['x'],5),round(r0['y'],5),"block right",round(d(obs,'target_block')['x']+tb['width'],5))
obs,*_=step(env,dx=0.05,dy=0.05,v=1.0)
r1=d(obs,'robot'); print("diagonal blocked-x step: dx",round(r1['x']-r0['x'],5),"dy",round(r1['y']-r0['y'],5))
print("block still held? block y",round(d(obs,'target_block')['y'],5),"robot y",round(r1['y'],5))
env.close()
