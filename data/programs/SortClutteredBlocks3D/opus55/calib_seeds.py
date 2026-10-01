from calib_util import *
env=make_env()
for s in range(12):
    obs,_=env.reset(seed=s); P={c:objpos(obs,c)[:2] for c in ['cube1','cube2','cube3','cube4']}
    best=max(P,key=lambda c:min(np.max(np.abs(P[c]-P[d])) for d in P if d!=c))
    iso=min(np.max(np.abs(P[best]-P[d])) for d in P if d!=best)
    print(s,best,P[best].round(3),'iso(chebyshev)',round(iso,3), rstate(obs)[0].round(3))
