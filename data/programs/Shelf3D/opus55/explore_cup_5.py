import sys, collections
from explore_cup_lib import *
from explore_cup_ik2 import ikl, Rp
env=make_env()
seed=int(sys.argv[1]); zc=sys.argv[2]; pdeg=float(sys.argv[3]); xb=float(sys.argv[4]); yw=float(sys.argv[5]); xin=float(sys.argv[6]) if len(sys.argv)>6 else 1.5
obs,info=env.reset(seed=seed)
cz=obs.get(obs.get_object_from_name('cupboard_1'),'z')
n=sorted(cubes(obs))[0]
zc=cz+float(zc[:-1]) if zc.endswith('r') else float(zc)
obs=pick(env,obs,n)
import helpers

R=Rp(np.radians(pdeg))
best=None
for q0 in [[-0.01,1.7,3.15,-1.95,-0.01,2.08,1.56],[-0.12,0.95,3.31,-2.45,-0.03,1.82,1.43],[-0.09,1.89,3.29,-1.08,-0.06,0.86,1.42],[-0.1,1.26,3.26,-2.19,0.,1.35,1.44]]:
    q,e1,e2=ikl(np.array([xb,0,zc]),R,q0)
    if best is None or e1+e2<best[1]+best[2]: best=(q,e1,e2)
q,e1,e2=best; print('ik',round(e1,3),round(e2,3),q.round(2))
b0=1.2-xb
obs,ok=drive2(env,obs,np.concatenate([[b0,yw,0],rstate(obs)[3:10]]),1.0,steps=300)
# go high intermediate then stage
qm=q.copy(); 
obs,ok=drive2(env,obs,np.concatenate([[b0,yw,0],q]),1.0,steps=400,tol=0.005)
print('LOGN',len(LOG));print('stage',ok,'cube',cubes(obs)[n][:3].round(3),'jerr',np.abs(wrap(q-rstate(obs)[3:10])).max().round(3))
b1=xin-xb
obs,ok=drive2(env,obs,np.concatenate([[b1,yw,0],q]),1.0,steps=300,tol=0.005)
print('LOGN',len(LOG));print('insert',ok,'base',rstate(obs)[:3].round(3),'cube',cubes(obs)[n][:3].round(3),'jerr',np.abs(wrap(q-rstate(obs)[3:10])).max().round(3))
obs=release(env,obs,15)
print('released cube',cubes(obs)[n][:3].round(3))
obs,ok=drive2(env,obs,np.concatenate([[b0-0.1,yw,0],q]),0.0,steps=300,tol=0.01)
obs=wait(env,obs,40)
c=cubes(obs)[n]
print('seed',seed,'cz',round(cz,3),'settled',c[:3].round(3),'z-cz',round(c[2]-cz-0.02,3),'quat',c[3:].round(2))
print('rewards',collections.Counter([l[0] for l in LOG]),'term',[i for i,l in enumerate(LOG) if l[1]][:3],'info',LOG[-1][3],'n',len(LOG))
ti=[i for i,l in enumerate(LOG) if l[1]]
for i in ti[:1]:
    print('first term step',i)
nz=[(i,l[0]) for i,l in enumerate(LOG) if l[0]!=-1.0]; print('nonm1',nz[:10])
