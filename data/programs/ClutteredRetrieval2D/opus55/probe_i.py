from probe_lib import *
exec(open('probe_h.py').read().split('for v in')[0])
o=grasp(11)
gx,gy,gt=bcenter(o,'target_region'); print('region center',round(gx,4),round(gy,4),'theta',gt)
bt=bcenter(o)[2]; want=bt+wrap(4*(gt-bt))/4  # nearest equiv mod pi/2
r=rob(o); o,ok=goto(o,th=r[2]+wrap(want-bt),vac=1); print('rotated',ok,bcenter(o))
def run_to(o,tx,ty,vac):
    for i in range(100):
        cx,cy,_=bcenter(o); dx=np.clip(tx-cx,-.02,.02); dy=np.clip(ty-cy,-.02,.02)
        if abs(dx)+abs(dy)<1e-6: break
        o,rew,term,tr,info=env.step(A(dx,dy,0,0,vac))
        if term or i%5==0: print(' step',i,'block c',np.round(bcenter(o)[:2],4),'rew',rew,'term',term,info)
        if term: break
    return o,term
o,term=run_to(o,gx,gy,1)
print('final held term',term)
if not term:
    o,rew,term,tr,info=env.step(A(0,0,0,0,0)); print('after release: rew',rew,'term',term)
