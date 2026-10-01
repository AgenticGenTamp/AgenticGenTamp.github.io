from grasp_utils import *
env = new_env()
obs, info = env.reset(seed=0)
# go to x=3.0 under shelf band pointing up
obs,ok,n,_ = drive_to(env,obs,2.97,2.0,np.pi/2,0.2,log=True)
obs,n = creep(env,obs,dy=0.01); r=rob(obs); print('up arm.2: y',r[1],'reach',2.625-r[1])
obs,ok,n,_ = drive_to(env,obs,2.97,2.0,np.pi/2,0.5,log=True)
obs,n = creep(env,obs,dy=0.01); r=rob(obs); print('up arm.5: y',r[1],'reach',2.625-r[1])
obs,n = creep(env,obs,darm=0.01); r=rob(obs); print('then arm creep', r)
# base alone: theta=0 (gripper sideways), arm .2
obs,ok,n,_ = drive_to(env,obs,2.97,2.0,0,0.2,log=True)
obs,n = creep(env,obs,dy=0.01); r=rob(obs); print('up theta0: y',r[1],'reach',2.625-r[1])
