from plib import *
A=0.15
def grasp_top(env,obs,name,g=0.005,arm=A):
    o=d(obs,name); cx=o['x']+o['width']/2; top=o['y']+o['height']
    obs,_=sety(env,obs,max(0.6,d(obs,'robot')['y']),vac=0)
    obs,_=setarm(env,obs,arm); obs,_=setx(env,obs,cx)
    obs,ok=sety(env,obs,top+g+arm+0.005)
    obs,*_=step(env,v=1.0)
    return obs
env=make_env(); obs,_=env.reset(seed=0)
for n in ['target_block','obstruction0','obstruction1','target_surface']:
    o=d(obs,n); print(n,"x",round(o['x'],4),"w",round(o['width'],4),"top",round(o['y']+o['height'],4))
print("== TASK5a: carry block down into table")
obs=grasp_top(env,obs,'target_block')
obs,_=setx(env,obs,0.3,vac=1)
delta=0.05
for i in range(40):
    prev=d(obs,'robot')['y']; obs,*_=step(env,dy=-delta,v=1.0)
    if abs(d(obs,'robot')['y']-prev)<1e-9:
        delta/=2
        if delta<1e-7: break
b=d(obs,'target_block'); r=d(obs,'robot')
print(f"  lowered: robot_y={r['y']:.5f} block_y={b['y']:.5f} (block bottom vs table 0.1) BLOCKED_BY_HELD_OBJ={'YES' if b['y']<0.1001 and r['y']>0.2001 else 'NO'}")
print("== TASK5b: carry block sideways into obstruction1 (x0.68 top0.2913)")
# lift so block bottom ~0.15 (inside obstruction1 vertical range 0.1..0.2913)
obs,_=sety(env,obs,d(obs,'robot')['y']+0.05,vac=1)
b=d(obs,'target_block'); print("  block y now",round(b['y'],4),"block right",round(b['x']+b['width'],4))
delta=0.05
for i in range(40):
    prev=d(obs,'robot')['x']; obs,*_=step(env,dx=delta,v=1.0)
    if abs(d(obs,'robot')['x']-prev)<1e-9:
        delta/=2
        if delta<1e-7: break
b=d(obs,'target_block'); print(f"  block right edge={b['x']+b['width']:.6f} obstruction1 left=0.68 -> BLOCKED={'YES' if b['x']+b['width']<0.6801 else 'NO'}")
print("== TASK4: release then grasp another object")
obs,*_=step(env,v=0.0)
obs,*_=step(env,v=0.0)
before=d(obs,'obstruction0')
obs,_=sety(env,obs,0.7)
obs=grasp_top(env,obs,'obstruction0')
obs,*_=step(env,dy=0.05,v=1.0)
after=d(obs,'obstruction0')
print("  obstruction0 moved dy",round(after['y']-before['y'],5),"REGRASP=",'YES' if after['y']-before['y']>0.04 else 'NO')
print("  target_block still at",round(d(obs,'target_block')['y'],4))
env.close()
