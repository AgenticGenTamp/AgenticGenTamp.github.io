from plib import *
A=0.15; DTH=0.19634
def setup(seed=0,g=0.0,arm=A):
    env=make_env(); obs,_=env.reset(seed=seed)
    tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
    obs,_=setarm(env,obs,arm); obs,_=setx(env,obs,cx)
    obs,_=sety(env,obs,top+g+arm+0.005)
    return env,obs
def rel(obs):
    r=d(obs,'robot'); b=d(obs,'target_block'); return (b['x']-r['x'],b['y']-r['y'],b['theta'])
def show(tag,obs,prev):
    c=rel(obs); print(f"  {tag}: rel_change=({c[0]-prev[0]:+.5f},{c[1]-prev[1]:+.5f}) blocktheta={c[2]:.4f} robot=({d(obs,'robot')['x']:.4f},{d(obs,'robot')['y']:.4f}) block=({d(obs,'target_block')['x']:.4f},{d(obs,'target_block')['y']:.4f})"); return c
print("== A: vac=1 combined with motion in same step (from contact)")
env,obs=setup(g=0.0)
b0=d(obs,'target_block')['y']
obs,*_=step(env,dy=0.05,v=1.0)
print("  block moved:",round(d(obs,'target_block')['y']-b0,5),"robot y",round(d(obs,'robot')['y'],4))
obs,*_=step(env,dy=0.05,v=1.0)
print("  next step block moved:",round(d(obs,'target_block')['y']-b0,5))
env.close()
print("== B: vac=1 static step then motions; then vac=0 static; then move vac=0")
env,obs=setup(g=0.0)
obs,*_=step(env,v=1.0); p=rel(obs)
obs,*_=step(env,dy=0.05,v=1.0); p=show("up vac1",obs,p)
obs,*_=step(env,v=0.0); p=show("static vac0",obs,p)
obs,*_=step(env,dy=0.05,v=0.0); p=show("up vac0",obs,p)
obs,*_=step(env,dy=0.05,v=0.0); p=show("up vac0 again",obs,p)
obs,*_=step(env,v=1.0); p=show("static vac1 (regrab?)",obs,p)
obs,*_=step(env,dy=0.05,v=1.0); p=show("up vac1",obs,p)
env.close()
print("== C: rotation while holding")
env,obs=setup(g=0.0)
obs,*_=step(env,v=1.0)
for i in range(6): obs,*_=step(env,dy=0.05,v=1.0)
p=rel(obs)
for i in range(4): obs,*_=step(env,dth=DTH,v=1.0)
p=show("rot +4*0.19634",obs,p)
env.close()
