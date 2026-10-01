from grasp_utils import *
env = new_env()
obs, info = env.reset(seed=1)
B='block1'
obs, ok = approach_and_grasp(env, obs, B, -1, log=True); print('grasp ok', ok, rob(obs).round(3))
o0=obs
obs,ok,n,_ = drive_to(env,obs,rob(obs)[0],rob(obs)[1]+0.3,rob(obs)[2],vac=1,log=True); print('lift',ok, blk(obs,B)-blk(o0,B))
obs,ok,n,_ = drive_to(env,obs,2.3,1.8,np.pi/2,vac=1,log=True); print('rotate up',ok, rob(obs).round(3), blk(obs,B).round(3))
r=rob(obs); cb=block_center(blk(obs,B)); dxc = 4.6005-cb[0]
obs,ok,n,_ = drive_to(env,obs,r[0]+dxc,2.0,np.pi/2,vac=1,log=True); print('under shelf',ok, rob(obs).round(3), block_corners(blk(obs,B)).round(3).tolist())
obs,n = creep(env,obs,dy=0.01,vac=1); print('creep up', rob(obs).round(4), block_corners(blk(obs,B)).round(3).tolist(), 'b0', blk(obs,'block0').round(3))
obs,n = creep(env,obs,darm=0.01,vac=1); print('creep arm', rob(obs).round(4), block_corners(blk(obs,B)).round(3).tolist(), 'b0', blk(obs,'block0').round(3))
obs,r_,term,trunc,info = env.step(act(vac=0)); print('release term',term, blk(obs,B).round(3))
