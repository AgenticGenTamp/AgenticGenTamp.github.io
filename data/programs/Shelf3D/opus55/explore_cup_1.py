import sys
from explore_cup_lib import *
env=make_env()
# args: seed bx then list of waypoints x,y,z (world)
seed=int(sys.argv[1]); bx=float(sys.argv[2]); wps=[list(map(float,w.split(','))) for w in sys.argv[3:]]
obs,info=env.reset(seed=seed)
n=sorted(cubes(obs))[0]
obs=pick(env,obs,n)
print('picked',cubes(obs)[n][:3].round(3))
obs,ok=carry_base(env,obs,bx,0.0); print('base',ok,rstate(obs)[:3].round(3),'cube',cubes(obs)[n][:3].round(3))
for w in wps:
    obs,ok,e,jerr=goto_cube(env,obs,w)
    print('wp',w,'ok',ok,'ikerr',round(e,4),'jerr',round(jerr,3),'cube',cubes(obs)[n][:3].round(3))
obs=release(env,obs); 
s=rstate(obs); pb=world_to_base(cubes(obs)[n][:3],s)
obs=wait(env,obs,40)
print('settled',cubes(obs)[n][:3].round(3), 'quat',cubes(obs)[n][3:].round(2))
import collections
print('rewards',collections.Counter([l[0] for l in LOG]),'term',any(l[1] for l in LOG),'lastinfo',LOG[-1][3], 'nsteps',len(LOG))
