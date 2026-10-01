from probe_dyn_lib import *
S=Sim()
def probe(start,d,seed=81,spd=0.004,n=200):
    S.reset(seed); S.goto_local(start); o0=S.obs.copy()
    rp0=w2l(o0,o0[16:18])
    for i in range(n):
        o=S.obs.copy(); S.step(rot(o[2])@np.array(d)*spd)
        if np.abs(S.obs[0:3]-o[0:3]).max()>1e-7:
            return w2l(o,o[16:18]), w2l(S.obs,S.obs[16:18]), rp0
    return None,None,rp0
o=S.reset(81); print(o[0:3],o[12:15])
for name,st,d in [('stem bottom x0',[0,-1.45],[0,1]),('stem bottom x.05',[0.05,-1.45],[0,1]),('stem bottom x.1',[0.1,-1.45],[0,1]),
                  ('bar top x0',[0,0.45],[0,-1]),('bar top x.5',[0.5,0.45],[0,-1]),('bar top x-.5',[-0.5,0.45],[0,-1]),
                  ('bar under x.4',[0.4,-0.5],[0,1]),('bar end +x',[0.9,0],[-1,0]),('bar end -x',[-0.9,0],[1,0]),
                  ('stem side +x y-.6',[0.6,-0.6],[-1,0]),('stem side -x y-.6',[-0.6,-0.6],[1,0])]:
    a,b,r0=probe(st,d); print(name,'start',r0,'contact at robot local',a)
