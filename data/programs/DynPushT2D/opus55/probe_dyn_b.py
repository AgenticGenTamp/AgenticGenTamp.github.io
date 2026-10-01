from probe_dyn_lib import *
import sys
S=Sim()
def exp(name,start,d,v,n,seed=81,verbose=False):
    S.reset(seed); S.goto_local(start)
    R=S.push(d,v,n)
    act=np.abs(R[:,:3]).max(1)>1e-7
    k=np.argmax(act)
    if verbose:
        for r in R[k:k+8]: print('   ',r[:3].round(5),'rp',r[3:5].round(4))
    Rs=R[k+1:]
    m=Rs[:,:3].mean(0)
    # instantaneous rotation center in local frame (of pre-step): p = (-dy/dth, dx/dth)
    rc=np.array([-m[1]/m[2],m[0]/m[2]]) if abs(m[2])>1e-6 else None
    print(f'{name:28s} v={v} first_contact_step={k} mean twist/step={m.round(5)} |dxy|/v={np.hypot(m[0],m[1])/v:.3f} rotC={None if rc is None else rc.round(3)} rp_first={R[k,3:5].round(3)} rp_last={R[-1,5:7].round(3)}')
    return R
if __name__=='__main__':
    for v in [0.01,0.03,0.0499]:
        exp('a stem bottom up x0',[0,-1.4],[0,1],v,int(0.6/v),verbose=(v==0.03))
    for x in [0,0.1,0.25,0.4,0.5]:
        for v in [0.01,0.0499]:
            exp(f'b bar top down x{x}',[x,0.3],[0,-1],v,int(0.5/v))
