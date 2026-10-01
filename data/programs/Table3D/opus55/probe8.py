from env_client import make_env
from kin import *
exec(open('probe3.py').read().split('env=make_env()')[0])
env=make_env()
for (x,y) in [(0.3,0.5),(0.0,0.6),(0.45,0.0),(0.75,0.0)]:
    obs,_=env.reset(seed=0)
    q=getq(obs); z=0.25
    while z>-0.5:
        qt,e=ik(q,np.array([x,y,z]),down_R(np.pi/2))
        obs,ok=step_to(env,obs,qt)
        if not ok or e>1e-3:
            print('stop at target z',round(z,3),'x,y',x,y,'ok',ok,'ikerr',e,'fk', fk(getq(obs))[:3,3]); break
        q=getq(obs); z-=0.01
env.close()
