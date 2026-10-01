from plib import *
A=0.15
# fine gap scan
for g in [0.012,0.014,0.015,0.016,0.018,0.019]:
    env=make_env(); obs,_=env.reset(seed=0)
    tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
    obs,_=setarm(env,obs,A); obs,_=setx(env,obs,cx)
    obs,ok=sety(env,obs,top+g+A+0.005)
    ry=d(obs,'robot')['y']
    obs,*_=step(env,v=1.0); b0=d(obs,'target_block')['y']
    obs,*_=step(env,dy=0.05,v=1.0); b1=d(obs,'target_block')['y']
    print(f"g={g:.4f} yok={ok} err={ry-(top+g+A+0.005):.2e} block_moved={b1-b0:.5f}")
    env.close()
# binary search contact height (descend)
env=make_env(); obs,_=env.reset(seed=0)
tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
obs,_=setarm(env,obs,A); obs,_=setx(env,obs,cx)
obs,_=sety(env,obs,top+0.03+A+0.005)
delta=0.02
for i in range(30):
    prev=d(obs,'robot')['y']
    obs,*_=step(env,dy=-delta)
    if abs(d(obs,'robot')['y']-prev)<1e-9:
        delta/=2
        if delta<1e-7: break
ry=d(obs,'robot')['y']
print(f"CONTACT descend: robot_y={ry:.7f} tip_formula={ry-(A+0.005):.7f} block_top={top:.7f} gap_at_contact={ry-(A+0.005)-top:.7f}")
# binary search contact by extending arm at fixed y
env2=make_env(); obs2,_=env2.reset(seed=0)
tb=d(obs2,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
obs2,_=setarm(env2,obs2,0.05); obs2,_=setx(env2,obs2,cx)
Y=top+0.30
obs2,_=sety(env2,obs2,Y)
Y=d(obs2,'robot')['y']
delta=0.1
for i in range(40):
    prev=d(obs2,'robot')['arm_joint']
    obs2,*_=step(env2,da=delta)
    if abs(d(obs2,'robot')['arm_joint']-prev)<1e-9:
        delta/=2
        if delta<1e-7: break
aj=d(obs2,'robot')['arm_joint']
print(f"CONTACT extend: robot_y={Y:.7f} arm={aj:.7f} tip={Y-(aj+0.005):.7f} gap={Y-(aj+0.005)-top:.7f}")
env.close(); env2.close()
