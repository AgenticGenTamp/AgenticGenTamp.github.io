from grasp_utils import *
env = new_env()
obs, info = env.reset(seed=1)
def C(n): return block_corners(blk(obs,n)).round(3).tolist()
obs, ok = approach_and_grasp(env, obs, 'block0', -1, arm=0.4, log=True); print('grasp b0', ok, rob(obs).round(3), C('block0'))
o0=obs
obs,n = creep(env,obs,dy=0.01,vac=1); print('b0 creep up', rob(obs).round(4), C('block0'))
obs,n = creep(env,obs,darm=0.01,vac=1); print('b0 creep arm', rob(obs).round(4), C('block0'))
# rotate to horizontal
r=rob(obs)
obs,ok,n,_ = drive_to(env,obs,r[0],r[1],np.pi/2,vac=1,log=True); print('rot horiz',ok, rob(obs).round(4), C('block0'))
obs,n = creep(env,obs,darm=0.01,vac=1); print('creep arm', rob(obs).round(4), C('block0'))
obs,n = creep(env,obs,dx=-0.01,vac=1); print('creep left', rob(obs).round(4), C('block0'))
obs,n = creep(env,obs,dx=0.01,vac=1); print('creep right', rob(obs).round(4), C('block0'))
obs,*_=env.step(act(vac=0)); print('released', C('block0'))
obs,ok,n,_ = drive_to(env,obs,rob(obs)[0],2.0,np.pi/2,0.2,log=True)
for B in ['block1','block2']:
    obs, ok = approach_and_grasp(env, obs, B, -1, log=True); print('grasp',B, ok, rob(obs).round(3))
    r=rob(obs)
    obs,ok,n,_ = drive_to(env,obs,r[0]-0.3*np.cos(r[2]),r[1]-0.3*np.sin(r[2]),r[2],vac=1,log=True)
    obs,ok,n,_ = drive_to(env,obs,3.0,1.8,np.pi/2,vac=1,log=True); print('rot up', ok, C(B))
    r=rob(obs); cb=block_center(blk(obs,B)); dxc = 4.6005-cb[0]
    obs,ok,n,_ = drive_to(env,obs,r[0]+dxc,2.0,np.pi/2,vac=1,log=True)
    obs,n = creep(env,obs,dy=0.01,vac=1); print('creep up', rob(obs).round(4), C(B))
    obs,n = creep(env,obs,darm=0.01,vac=1); print('creep arm', rob(obs).round(4), C(B))
    obs,rw,term,trunc,info=env.step(act(vac=1)); print('term before release', term)
    obs,rw,term,trunc,info=env.step(act(vac=0)); print('released term', term, rw)
    if term: break
    obs,ok,n,_ = drive_to(env,obs,rob(obs)[0],1.9,np.pi/2,0.2,log=True)
