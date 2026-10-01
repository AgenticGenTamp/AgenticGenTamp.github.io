import concurrent.futures
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))

def trial(spec):
    axis,delta,n,base_x,base_y=spec
    e=make_env(); s,info=e.reset(seed=0,options={'object_count':1})
    p=GeneratedApproach(e.action_space,e.observation_space,{}) ; p.reset(s,info)
    for step in range(230):
        a=p.get_action(s); s,r,t,tr,i=e.step(a)
        if p.phase=='tool_stage' and p.tool_attempt==1: break
    # local seating move while keeping hold
    for k in range(n):
        a=np.zeros(11,np.float32); a[axis]=delta; a[0]+=base_x; a[1]+=base_y; a[10]=-1
        s,r,t,tr,i=e.step(a)
        if t: break
    # release and wait
    if not t:
        for k in range(6):
            a=np.zeros(11,np.float32); a[10]=1
            s,r,t,tr,i=e.step(a)
            if t: break
    out=(spec,t,step,v(s,'cube0','pose_x'),v(s,'cube0','pose_y'),v(s,'cube0','pose_z'),v(s,'box0','pose_z'))
    e.close(); return out

if __name__ == '__main__':
    specs=[(3+i,d,n,0.,0.) for i in range(7) for d in (-.2,-.1,.1,.2) for n in (1,3,6)]
    specs += [(3,0.,n,x,y) for x,y in [(.05,0),(-.05,0),(0,.05),(0,-.05),(.05,.05),(.05,-.05),(-.05,.05),(-.05,-.05)] for n in (1,3,6)]
    specs.insert(0,(3,0.,0,0.,0.))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for z in ex.map(trial,specs):
            if z[1]: print('SUCCESS',z,flush=True)
