from probe_dyn_lib import *
import sys
rng=np.random.default_rng(int(sys.argv[1]))
seeds=[int(s) for s in sys.argv[2].split(',')]
nexp=int(sys.argv[3])
S=Sim(); data=[]
def faces(w,L,lv):
    return [((-L/2,0),(L/2,0),(0,1)),((L/2,-w),(L/2,0),(1,0)),((-L/2,-w),(-L/2,0),(-1,0)),
            ((w/2,-w),(L/2,-w),(0,-1)),((-L/2,-w),(-w/2,-w),(0,-1)),
            ((w/2,-w-lv),(w/2,-w),(1,0)),((-w/2,-w-lv),(-w/2,-w),(-1,0)),((-w/2,-w-lv),(w/2,-w-lv),(0,-1))]
def dist_T(q,w,L,lv):
    def dr(q,x0,x1,y0,y1):
        dx=max(x0-q[0],0,q[0]-x1); dy=max(y0-q[1],0,q[1]-y1); return np.hypot(dx,dy)
    return min(dr(q,-L/2,L/2,-w,0),dr(q,-w/2,w/2,-w-lv,-w))
for e in range(nexp):
    seed=seeds[e%len(seeds)]; o=S.reset(seed); w,L,lv=o[12:15]
    F=faces(w,L,lv); lens=[np.hypot(f[1][0]-f[0][0],f[1][1]-f[0][1]) for f in F]
    i=rng.choice(len(F),p=np.array(lens)/sum(lens)); a,b,n=F[i]; s=rng.uniform(0.03,0.97)
    p=np.array(a)+s*(np.array(b)-np.array(a)); n=np.array(n,float)
    ang=rng.uniform(-1.2,1.2); t=np.array([-n[1],n[0]])
    ud=-n*np.cos(ang)+t*np.sin(ang)
    start=p+n*0.1-ud*0.12
    if dist_T(start,w,L,lv)<0.105: continue
    mv=S.goto_local(start)
    if mv>1e-9: continue
    v=rng.choice([0.01,0.02,0.03,0.0499]); nst=int(0.12/v)+int(rng.integers(4,10))
    for k in range(nst):
        o=S.obs.copy(); u=rot(o[2])@ud*v
        o2=S.step(u)
        data.append(np.r_[seed,e,k,o[0:3],o[16:18],u,o2[0:3],o2[16:18],w,L,lv,v,ang])
np.save(f'probe_dyn_data_{sys.argv[1]}.npy',np.array(data)); print(len(data))
