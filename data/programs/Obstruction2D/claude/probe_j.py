from plib import *
A=0.15
print("== TASK4 regrasp: grasp target_block, release on table, grasp again")
env=make_env(); obs,_=env.reset(seed=0)
tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
obs,_=setarm(env,obs,A); obs,_=setx(env,obs,cx); obs,_=sety(env,obs,top+0.005+A+0.005)
obs,*_=step(env,v=1.0)
for i in range(3): obs,*_=step(env,dy=0.05,v=1.0)
obs,_=setx(env,obs,0.25,vac=1)
delta=0.05
for i in range(30):
    prev=d(obs,'robot')['y']; obs,*_=step(env,dy=-delta,v=1.0)
    if abs(d(obs,'robot')['y']-prev)<1e-9:
        delta/=2
        if delta<1e-7: break
print("  placed block at",{k:round(v,4) for k,v in d(obs,'target_block').items() if k in('x','y')})
obs,*_=step(env,v=0.0); obs,*_=step(env,v=0.0)
obs,_=sety(env,obs,0.6)
print("  block after release+retreat y=",round(d(obs,'target_block')['y'],5))
tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
obs,_=setx(env,obs,cx); obs,_=sety(env,obs,top+0.005+A+0.005)
obs,*_=step(env,v=1.0)
b0=d(obs,'target_block')['y']
obs,*_=step(env,dy=0.05,v=1.0)
print("  REGRASP same object dy:",round(d(obs,'target_block')['y']-b0,5))
# now release and grasp a DIFFERENT object (obstruction0) far away? seed0 crowded; just report
env.close()
print("== TASK6 multi-object grasp, seed 13")
env=make_env(); obs,_=env.reset(seed=13)
o0=d(obs,'obstruction0'); o1=d(obs,'obstruction1')
print("  o0",{k:round(v,4) for k,v in o0.items() if k in('x','y','width','height')})
print("  o1",{k:round(v,4) for k,v in o1.items() if k in('x','y','width','height')})
seam=(o0['x']+o0['width']+o1['x'])/2
tops=max(o0['y']+o0['height'],o1['y']+o1['height'])
obs,_=setarm(env,obs,A)
obs,okx=setx(env,obs,seam)
print("  seam target",round(seam,4),"robot x",round(d(obs,'robot')['x'],4),okx)
obs,oky=sety(env,obs,tops+0.004+A+0.005)
print("  robot y",round(d(obs,'robot')['y'],4),"tip",round(d(obs,'robot')['y']-A-0.005,4),"o0top",round(o0['y']+o0['height'],4),"o1top",round(o1['y']+o1['height'],4))
obs,*_=step(env,v=1.0)
b0=[d(obs,'obstruction0')['y'],d(obs,'obstruction1')['y']]
obs,*_=step(env,dy=0.05,v=1.0)
b1=[d(obs,'obstruction0')['y'],d(obs,'obstruction1')['y']]
print("  MULTI: o0 dy",round(b1[0]-b0[0],5),"o1 dy",round(b1[1]-b0[1],5))
env.close()
print("== TASK6 vac=1 in free space")
env=make_env(); obs,_=env.reset(seed=0)
obs,_=setx(env,obs,0.25); obs,_=sety(env,obs,0.6)
for i in range(3): obs,r,t,tr,info=step(env,v=1.0)
print("  ok, reward",r,"term",t,"robot",{k:round(v,4) for k,v in d(obs,'robot').items() if k in('x','y','vacuum')})
obs,*_=step(env,dx=0.05,v=1.0); obs,*_=step(env,dy=-0.05,v=1.0)
print("  moves still work:",{k:round(v,4) for k,v in d(obs,'robot').items() if k in('x','y')})
env.close()
