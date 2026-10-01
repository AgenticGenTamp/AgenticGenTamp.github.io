from grasp_utils import *
env = new_env()
B='block3'
def rel(obs):
    r=rob(obs); b=blk(obs,B)
    # block pose in robot frame
    c,s=np.cos(r[2]),np.sin(r[2]); d=b[:2]-r[:2]
    return np.array([c*d[0]+s*d[1], -s*d[0]+c*d[1], wrap(b[2]-r[2])])
obs, info = env.reset(seed=0)
obs,ok,n,_ = drive_to(env,obs,2.97,1.35,np.pi/2,0.2,log=True)
obs,ok,n,_ = drive_to(env,obs,3.84,1.35,np.pi/2,0.2,log=True); print(ok, rob(obs))
obs,n = creep(env,obs,dy=0.01); r=rob(obs); print('contact y',r[1],'tip',r[1]+0.21, 'blk',blk(obs,B))
obs,*_ = env.step(act(vac=1)); print('vac on', rob(obs), blk(obs,B), 'rel',rel(obs))
tests=[('down',act(dy=-0.05,vac=1)),('left',act(dx=-0.05,vac=1)),('rot',act(dth=0.1,vac=1)),('arm+',act(darm=0.1,vac=1)),('rot-big',act(dth=-0.196,vac=1)),('combo',act(-.03,-.04,.1,.05,1))]
for name,a in tests:
    obs,*_=env.step(a); print(name,'rob',rob(obs).round(4),'rel',rel(obs).round(5))
b0=blk(obs,B)
obs,*_ = env.step(act(vac=0)); print('release: blk d', blk(obs,B)-b0)
obs,*_ = env.step(act(dy=-0.05)); print('after release move: blk d', blk(obs,B)-b0, rob(obs).round(3))
