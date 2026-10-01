import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
env = make_env(); obs, info = env.reset(seed=1); S = Sim(env, obs)
name = sorted(S.rods())[0]; pick_rod(S, name)
Rv = np.column_stack([[0,1,0],[0,0,1],[1,0,0]])
bt=np.array([1.1, -0.1, 0.0]); S.goto(bt=bt, steps=100)
b=S.base(); print('base', np.round(b,3), 'q', np.round(S.q(),2))
qt,e1,e2 = ik(S.q(), world_to_arm(b, np.array([1.45,-0.1,0.365])), R_world_to_arm(b,Rv), tool=TOOL)
print('qt', np.round(qt,2), e1, e2)
for k in range(8):
    S.goto(qt=qt, steps=25)
    print(np.round(S.q(),2), 'rod', np.round(S.rods()[name][:3],2))
env.close()
