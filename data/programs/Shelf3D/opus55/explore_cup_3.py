from explore_cup_lib import *
env=make_env()
obs,info=env.reset(seed=1); n='cube1' if 'cube1' in cubes(obs) else sorted(cubes(obs))[0]
obs=pick(env,obs,n)
Rh=np.array([[0,0,1],[1,0,0],[0,1,0.]])
rows=[]
for R,p in [(Rdown(0),np.array([0.45,0,0.15])),(Rh,np.array([0.35,0,0.2])),(Rh,np.array([0.4,0,0.5])),(Rdown(0),np.array([0.5,0,0.4]))]:
    s=rstate(obs); q,e1,e2=ik(p,R,s[3:10])
    obs,ok=drive2(env,obs,np.concatenate([s[:3],q]),1.0,steps=300,tol=0.005)
    obs=wait(env,obs,10,1.0)
    s=rstate(obs); T=fk_arm(s[3:10]); c=world_to_base(cubes(obs)[n][:3],s)
    print(ok,round(e1,4),'flange',T[:3,3].round(3),'z',T[:3,2].round(2),'cube_b',c.round(3))
