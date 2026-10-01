from plib import *
DTH=0.19634
env=make_env(); obs,_=env.reset(seed=0)
tb=d(obs,'target_block'); left=tb['x']; print("block",{k:round(v,4) for k,v in tb.items() if k in('x','y','width','height')})
obs,_=setarm(env,obs,0.05)
obs,_=move_to(env,obs,'theta',0.0,2,DTH)
obs,_=setx(env,obs,0.15)
obs,ok=sety(env,obs,0.2)
print("robot",{k:round(v,4) for k,v in d(obs,'robot').items() if k in('x','y','theta','arm_joint')})
# approach right until blocked
delta=0.05
for i in range(40):
    prev=d(obs,'robot')['x']; obs,*_=step(env,dx=delta)
    if abs(d(obs,'robot')['x']-prev)<1e-9:
        delta/=2
        if delta<1e-7: break
r=d(obs,'robot'); tip=r['x']+r['arm_joint']+0.005
print(f"contact x={r['x']:.6f} tip={tip:.6f} block_left={left:.6f} gap={left-tip:.6f}")
b0=d(obs,'target_block')
obs,*_=step(env,v=1.0)
obs,*_=step(env,dx=-0.05,v=1.0)
obs,*_=step(env,dy=0.05,v=1.0)
b1=d(obs,'target_block')
print("SIDE GRASP: block moved dx",round(b1['x']-b0['x'],5),"dy",round(b1['y']-b0['y'],5))
env.close()
