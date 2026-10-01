import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
env = make_env(); obs, info = env.reset(seed=1)
S = Sim(env, obs)
S.goto(bt=np.array([0.0,-0.5,0.0]), steps=40)   # move away from rods
print('base', np.round(S.base(),3))
Rd=np.array([[1,0,0],[0,-1,0],[0,0,-1.]])
q0=S.q()
# go above (0.5,0, z) in arm frame (tool=0) and lower slowly
for z in [0.0,-0.1,-0.2,-0.25,-0.3,-0.35,-0.4,-0.45,-0.5]:
    qt,e1,e2 = ik(S.q(), np.array([0.5,0,z]), Rd, tool=0.0)
    S.goto(qt=qt, steps=40)
    qa=S.q(); print(z, 'ik_err',round(e1,4), 'achieved z', round(fk(qa)[2,3],3), 'track err', np.round(np.abs(qa-qt).max(),3))
env.close()
