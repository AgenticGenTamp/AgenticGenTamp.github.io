from plib import *
A=0.15
for g in [0.0,0.001,0.002,0.005,0.01,0.02,0.03]:
    env=make_env(); obs,_=env.reset(seed=0)
    tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
    obs,ok1=setarm(env,obs,A)
    obs,ok2=setx(env,obs,cx)
    ty=top+g+A+0.005
    obs,ok3=sety(env,obs,ty)
    r=d(obs,'robot')
    obs,*_=step(env,v=1.0)
    b0=d(obs,'target_block')['y']
    ry0=d(obs,'robot')['y']
    obs,*_=step(env,dy=0.05,v=1.0)
    b1=d(obs,'target_block')['y']; ry1=d(obs,'robot')['y']
    print(f"g={g:.4f} armok={ok1} xok={ok2} yok={ok3} ry={r['y']:.6f} target={ty:.6f} robot_moved={ry1-ry0:.4f} block_moved={b1-b0:.5f} GRASP={'YES' if b1-b0>0.04 else 'NO'}")
    env.close()
