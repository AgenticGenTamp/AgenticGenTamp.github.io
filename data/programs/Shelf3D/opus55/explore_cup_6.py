import sys, collections
from explore_cup_lib import *
from explore_cup_ik2 import ikl, Rp
env=make_env()
seed=int(sys.argv[1]); plan=[(float(a.split(',')[0]),float(a.split(',')[1])) for a in sys.argv[2:]]  # (zrel, y)
obs,info=env.reset(seed=seed)
cz=obs.get(obs.get_object_from_name('cupboard_1'),'z')
names=sorted(cubes(obs)); print('cubes',{k:v[:3].round(2) for k,v in cubes(obs).items()})
Q0=[[-0.01,1.7,3.15,-1.95,-0.01,2.08,1.56],[-0.12,0.95,3.31,-2.45,-0.03,1.82,1.43]]
for n,(zr,yw) in zip(names,plan):
    obs=pick(env,obs,n)
    zc=cz+zr; xb=0.75
    q=min([ikl(np.array([xb,0,zc]),Rp(0),q0) for q0 in Q0],key=lambda r:r[1]+r[2])[0]
    b0=1.2-xb
    obs,ok=drive2(env,obs,np.concatenate([[b0,yw,0],rstate(obs)[3:10]]),1.0,steps=300)
    obs,ok=drive2(env,obs,np.concatenate([[b0,yw,0],q]),1.0,steps=400,tol=0.005)
    obs,ok=drive2(env,obs,np.concatenate([[1.5-xb,yw,0],q]),1.0,steps=300,tol=0.005)
    obs=release(env,obs,15)
    obs,ok=drive2(env,obs,np.concatenate([[b0-0.1,yw,0],q]),0.0,steps=300,tol=0.01)
    obs,ok=drive2(env,obs,np.concatenate([[b0-0.1,yw,0],HOME]),0.0,steps=300,tol=0.01)
    print('placed',n,'len',len(LOG),{k:(v[:3]-[0,0,cz+0.02]).round(3) for k,v in cubes(obs).items()},'term_now',LOG[-1][1])
obs=wait(env,obs,20)
print('rewards',collections.Counter([l[0] for l in LOG]),'term',[i for i,l in enumerate(LOG) if l[1]][:3],'final term',LOG[-1][1])
