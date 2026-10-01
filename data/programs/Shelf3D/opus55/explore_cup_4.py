import sys, collections
from explore_cup_lib import *
env=make_env(); GRIP=[1.0]
seed=int(sys.argv[1]); toks=sys.argv[2:]
obs,info=env.reset(seed=seed)
n=sorted(cubes(obs))[0]
obs=pick(env,obs,n)
for t in toks:
    if t.startswith('b='):
        bx,by=map(float,t[2:].split(',')); obs,ok=carry_base(env,obs,bx,by); print('base',ok,rstate(obs)[:3].round(3),'cube',cubes(obs)[n][:3].round(3))
    elif t.startswith('r'):
        z0=float(t[1:]); st=rstate(obs)
        q1,_,_=ik(np.array([0.4,0,0.49]),Rh,st[3:10])
        obs,ok=drive2(env,obs,np.concatenate([st[:3],q1]),1.0,steps=300,tol=0.005)
        q2=q1;e=0
        print('reorient',ok,round(e,4),'cube',cubes(obs)[n][:3].round(3),'q',q2.round(2))
    elif t=='open':
        obs=release(env,obs); print('open cube',cubes(obs)[n][:3].round(3))
    else:
        R=Rdown(0) if t[0]=='d' else Rh
        w=list(map(float,t.lstrip('dh').split(',')))
        g=0.0 if len(LOG) and t[0] in 'dh' and False else 1.0
        obs,ok,e,jerr=goto_h(env,obs,w,grip=GRIP[0],R=R)
        print(t,'ok',ok,'ikerr',round(e,4),'jerr',round(jerr,3),'cube',cubes(obs)[n][:3].round(3))
    if t=='open': GRIP[0]=0.0
obs=wait(env,obs,40)
print('settled',cubes(obs)[n][:3].round(3), 'quat',cubes(obs)[n][3:].round(2))
print('rewards',collections.Counter([l[0] for l in LOG]),'term',any(l[1] for l in LOG),'trunc',any(l[2] for l in LOG),'lastinfo',LOG[-1][3], 'nsteps',len(LOG))
