from grasp_utils import *
env = new_env()
obs, info = env.reset(seed=1)
def C(n, o=None): return block_corners(blk(o if o is not None else obs,n)).round(3).tolist()
def insert(obs, B, side, waypoint=None, arm_up=True):
    if waypoint: obs,*_ = drive_to(env,obs,*waypoint,0.2,log=True)
    obs, ok = approach_and_grasp(env, obs, B, side, log=True); print('grasp',B, ok, rob(obs).round(3))
    r=rob(obs)
    obs,ok,n,_ = drive_to(env,obs,r[0]-0.3*np.cos(r[2]),r[1]-0.3*np.sin(r[2]),r[2],vac=1,log=True)
    obs,ok,n,_ = drive_to(env,obs,rob(obs)[0],1.8,np.pi/2,vac=1,log=True); print('rot up', ok, C(B,obs))
    r=rob(obs); cb=block_center(blk(obs,B)); dxc = 4.6005-cb[0]
    obs,ok,n,_ = drive_to(env,obs,r[0]+dxc,2.0,np.pi/2,vac=1,log=True)
    obs,n = creep(env,obs,dy=0.01,vac=1); print('creep up', rob(obs).round(4), C(B,obs))
    if arm_up: obs,n = creep(env,obs,darm=0.01,vac=1); print('creep arm', rob(obs).round(4), C(B,obs))
    return obs
obs, ok = approach_and_grasp(env, obs, 'block0', -1, arm=0.4)
obs,n = creep(env,obs,dy=0.01,vac=1); obs,n = creep(env,obs,darm=0.01,vac=1)
obs,*_=env.step(act(vac=0)); obs,*_ = drive_to(env,obs,rob(obs)[0],2.0,np.pi/2,0.2)
obs = insert(obs,'block1',-1)
obs,*_=env.step(act(vac=0)); obs,*_ = drive_to(env,obs,rob(obs)[0],2.0,np.pi/2,0.2)
obs = insert(obs,'block2',+1, waypoint=(3.9,1.8,np.pi), arm_up=False)
obs,*_=env.step(act(darm=-0.05,vac=1)); print('ymin', min(p[1] for p in C('block2',obs)))
for k in range(40):
    obs,rw,term,trunc,info=env.step(act(darm=0.002,vac=1)); c=C('block2',obs)
    print('arm', round(rob(obs)[3],4),'blk ymin', min(p[1] for p in c), 'term',term)
    if term: break
