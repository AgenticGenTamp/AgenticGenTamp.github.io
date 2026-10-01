import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
env=make_env(); obs,info=env.reset(seed=1); S=Sim(env,obs)
name='cuboid_1'
print(pick_rod(S,name), len(S.rew))
RV=np.array([[0,0,1],[0,1,0],[-1,0,0.]])
b=np.array([1.3,0,0])
for p in [[1.5,0,0.5],[1.6,0,0.7],[1.7,0,0.8]]:
    e,n=move_ee_world(S,np.array(p,float),RV,bt=b,steps=150)
    r=S.rods()[name]; print(p,'ikerr',np.round(e,3),n,'gp',np.round(grasp_point_world(S),3),'rod',np.round(r,3),'base',np.round(S.base(),3),len(S.rew))
print('nonm1',[(i,x) for i,x in enumerate(S.rew) if x!=-1.0])
env.close()
