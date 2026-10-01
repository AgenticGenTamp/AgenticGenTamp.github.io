from plib import *
A=0.15
def setup(g,seed=0,arm=A):
    env=make_env(); obs,_=env.reset(seed=seed)
    tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
    obs,_=setarm(env,obs,arm); obs,_=setx(env,obs,cx); obs,_=sety(env,obs,top+g+arm+0.005)
    return env,obs
for g in [0.0,0.005]:
    print("=== gap",g)
    env,obs=setup(g)
    seq=[(0,1),(0.05,1),(0.05,0),(0.05,0),(0.05,1),(0.05,1)]
    print(f"  start robot_y={d(obs,'robot')['y']:.4f} block_y={d(obs,'target_block')['y']:.4f}")
    for dy,v in seq:
        obs,*_=step(env,dy=dy,v=v)
        print(f"  dy={dy} vac={v} -> robot_y={d(obs,'robot')['y']:.4f} block_y={d(obs,'target_block')['y']:.4f} robvacfeat={d(obs,'robot')['vacuum']}")
    env.close()
