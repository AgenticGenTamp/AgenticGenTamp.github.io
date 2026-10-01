from plib import *
A=0.15
def setup(seed=0,g=0.0,arm=A):
    env=make_env(); obs,_=env.reset(seed=seed)
    tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
    obs,_=setarm(env,obs,arm); obs,_=setx(env,obs,cx)
    obs,_=sety(env,obs,top+g+arm+0.005)
    return env,obs
# bisect grasp gap threshold
lo,hi=0.014,0.015
for i in range(10):
    g=(lo+hi)/2
    env,obs=setup(g=g)
    obs,*_=step(env,v=1.0); b0=d(obs,'target_block')['y']
    obs,*_=step(env,dy=0.05,v=1.0); b1=d(obs,'target_block')['y']
    if b1-b0>0.04: lo=g
    else: hi=g
    env.close()
print(f"GAP THRESHOLD: works<= {lo:.6f}, fails>= {hi:.6f}")
# TASK2: does vac need to be held?
env,obs=setup(g=0.005)
obs,*_=step(env,v=1.0)
obs,*_=step(env,dy=0.05,v=1.0)
b_before=d(obs,'target_block')['y']
obs,*_=step(env,dy=0.05,v=0.0)   # move with vac off
b_after=d(obs,'target_block')['y']; r=d(obs,'robot')
print(f"TASK2 move with vac=0 after grasp: robot dy ok, block moved {b_after-b_before:.5f}, robot vacuum feat={r['vacuum']}")
# then turn vac back on and move
obs,*_=step(env,dy=0.05,v=1.0)
print("  after re-enabling vac and moving up: block moved", round(d(obs,'target_block')['y']-b_after,5))
env.close()
# vac=1 every step, many moves
env,obs=setup(g=0.005)
obs,*_=step(env,v=1.0)
b0=d(obs,'target_block'); r0=d(obs,'robot')
for i in range(5): obs,*_=step(env,dy=0.05,v=1.0)
for i in range(5): obs,*_=step(env,dx=-0.05,v=1.0)
for i in range(3): obs,*_=step(env,dth=0.19635,v=1.0)
b1=d(obs,'target_block'); r1=d(obs,'robot')
print("TASK2b held rigid: d(block-robot) x",round((b1['x']-r1['x'])-(b0['x']-r0['x']),5),
      "y",round((b1['y']-r1['y'])-(b0['y']-r0['y']),5),"block theta",round(b1['theta'],5),"robot theta",round(r1['theta'],5))
env.close()
