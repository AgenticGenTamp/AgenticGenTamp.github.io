from plib import *
DTH=0.19634
def d2(obs,n): return d(obs,n)
# --- how low can the robot go in free space (table collision)?
env=make_env(); obs,_=env.reset(seed=0)
obs,_=setarm(env,obs,0.0); obs,_=setx(env,obs,0.25)
delta=0.05
for i in range(40):
    prev=d(obs,'robot')['y']; obs,*_=step(env,dy=-delta)
    if abs(d(obs,'robot')['y']-prev)<1e-9:
        delta/=2
        if delta<1e-7: break
print("arm=0 min robot y (free space x=0.25):",round(d(obs,'robot')['y'],6),"=> tip",round(d(obs,'robot')['y']-0.005,6))
env.close()
# with theta=0 (arm horizontal) how low
env=make_env(); obs,_=env.reset(seed=0)
obs,_=setarm(env,obs,0.05); obs,_=setx(env,obs,0.25)
obs,_=move_to(env,obs,'theta',0.0,2,DTH)
print("theta now",round(d(obs,'robot')['theta'],5))
delta=0.05
for i in range(40):
    prev=d(obs,'robot')['y']; obs,*_=step(env,dy=-delta)
    if abs(d(obs,'robot')['y']-prev)<1e-9:
        delta/=2
        if delta<1e-7: break
print("theta=0 arm=0.05 min robot y:",round(d(obs,'robot')['y'],6))
env.close()
