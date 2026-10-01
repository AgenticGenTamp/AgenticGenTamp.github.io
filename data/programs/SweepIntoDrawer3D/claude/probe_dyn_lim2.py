import numpy as np, threading, json
from env_client import make_env
R={}
def run(name,fn):
    try:
        env=make_env(); o,_=env.reset(seed=0); R[name]=fn(env,np.asarray(o,float)); env.close()
    except Exception as e: R[name]='ERR '+repr(e)
def A(dx=0,dy=0,dyaw=0,j=None,g=0.0):
    a=np.zeros(11); a[0]=dx;a[1]=dy;a[2]=dyaw
    if j:
        for k,v in j.items(): a[3+k]=v
    a[10]=g; return a
def roll(env,a,n):
    S=[];RW=[]
    for _ in range(n):
        o,r,t,tr,_=env.step(a); S.append(np.asarray(o,float)[125:147]); RW.append(r)
    return np.array(S),np.array(RW)
def mk_lim(k,s):
    def f(env,o0):
        S,RW=roll(env,A(j={k:s*0.1}),220)
        q=S[:,3+k]; d=np.abs(np.diff(q))
        stall=None
        for i in range(len(d)-4):
            if np.all(d[i:i+5]<2e-4): stall=i; break
        return {'q0':round(float(o0[128+k]),4),'qfinal':round(float(q[-1]),4),
                'stall_step':stall,'q_at_stall':(round(float(q[stall]),4) if stall is not None else None),
                'vel_at_stall':(round(float(S[stall][3+k+11]),4) if stall is not None else None),
                'rew':sorted(set(np.round(RW,3).tolist()))}
    return f
def mk_lin(c):
    def f(env,o0):
        S,_=roll(env,A(j={0:c}),25)
        inc=np.diff(np.concatenate([[o0[128]],S[:,3]]))
        return {'cmd':c,'inc_first5':np.round(inc[:5],5).tolist(),'ss_inc':round(float(np.mean(inc[-5:])),5),
                'ratio':round(float(np.mean(inc[-5:])/c),4),'vel_ss':round(float(np.mean(S[-5:,14])),4)}
    return f
def f_basefree(env,o0):
    out={}
    # +x free space (away from table)
    S,_=roll(env,A(dx=0.1),6); out['px_inc']=np.round(np.diff(np.concatenate([[o0[125]],S[:,0]])),5).tolist()
    S,_=roll(env,A(),3)
    S,_=roll(env,A(dy=0.1),6); out['py_inc']=np.round(np.diff(np.concatenate([[S[0,1]],S[:,1]])),5).tolist()
    S2,_=roll(env,A(),3)
    S,_=roll(env,A(dy=-0.1),6); out['ny_inc']=np.round(np.diff(np.concatenate([[S2[-1,1]],S[:,1]])),5).tolist()
    S2,_=roll(env,A(),3)
    S,_=roll(env,A(dx=0.05),6); out['px05_inc']=np.round(np.diff(np.concatenate([[S2[-1,0]],S[:,0]])),5).tolist()
    S2,_=roll(env,A(),3)
    S,_=roll(env,A(dx=0.1,dy=0.1),6); out['diag_inc']=np.round(np.diff(S[:,:2],axis=0),5).tolist()
    return out
def f_overcmd(env,o0):
    S,_=roll(env,A(j={0:0.5}),8)  # beyond action range
    return {'inc':np.round(np.diff(np.concatenate([[o0[128]],S[:,3]])),5).tolist()}
jobs=[('basefree',f_basefree),('over',f_overcmd)]
for c in [0.025,0.05,0.1]: jobs.append(('lin%g'%c,mk_lin(c)))
for k in range(7):
    jobs.append(('LP%d'%(k+1),mk_lim(k,1))); jobs.append(('LN%d'%(k+1),mk_lim(k,-1)))
th=[]
for n,f in jobs:
    t=threading.Thread(target=run,args=(n,f)); t.start(); th.append(t)
for t in th: t.join()
json.dump(R,open('res_lim2.json','w'),indent=1); print('done',len(R))
