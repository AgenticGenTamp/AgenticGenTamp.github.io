import sys, collections
from explore_cup_lib import *
env=make_env()
seed=int(sys.argv[1]); toks=sys.argv[2:]
obs,info=env.reset(seed=seed)
n=sorted(cubes(obs))[0]
obs=pick(env,obs,n)
for t in toks:
    if t.startswith('b='):
        bx,by=map(float,t[2:].split(',')); obs,ok=carry_base(env,obs,bx,by); print('base',ok,rstate(obs)[:3].round(3),'cube',cubes(obs)[n][:3].round(3))
    else:
        w=list(map(float,t.split(',')))
        obs,ok,e,jerr=goto_cube(env,obs,w)
        print('wp',w,'ok',ok,'ikerr',round(e,4),'jerr',round(jerr,3),'cube',cubes(obs)[n][:3].round(3))
obs=release(env,obs)
obs=wait(env,obs,40)
print('settled',cubes(obs)[n][:3].round(3), 'quat',cubes(obs)[n][3:].round(2))
print('rewards',collections.Counter([l[0] for l in LOG]),'term',any(l[1] for l in LOG),'lastinfo',LOG[-1][3], 'nsteps',len(LOG))
