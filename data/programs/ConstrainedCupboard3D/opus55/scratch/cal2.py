import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
env = make_env(); obs, info = env.reset(seed=1)
S = Sim(env, obs)
S.goto(bt=np.array([0.0,-0.5,0.0]), steps=40)
S.goto(bt=np.array([0.0,-0.339,0.0]), steps=40)
print('base', np.round(S.base(),3), 'rod0', np.round(S.rods()['cuboid_0'][:3],3))
Rd=np.array([[1,0,0],[0,-1,0],[0,0,-1.]])
qt,_,_ = ik(S.q(), np.array([0.3,0,-0.1]), Rd); S.goto(qt=qt, steps=60)
qt,_,_ = ik(S.q(), np.array([0.3,0,-0.24]), Rd); S.goto(qt=qt, steps=60)
r0 = S.rods()['cuboid_0'][:3]
for x in np.arange(0.3, 0.9, 0.01):
    qt,_,_ = ik(S.q(), np.array([x,0,-0.24]), Rd); S.goto(qt=qt, steps=15, tol=0.003)
    r = S.rods()['cuboid_0'][:3]; b=S.base()
    if np.linalg.norm(r-r0)>0.003:
        print('moved at flange x', round(x,3), 'actual flange', np.round(fk(S.q())[:3,3],3), 'rod', np.round(r,3), 'base', np.round(b,3)); break
env.close()
