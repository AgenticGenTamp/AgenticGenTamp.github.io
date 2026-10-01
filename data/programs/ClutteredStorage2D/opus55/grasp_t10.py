from grasp_utils import *
env = new_env()
cnt={}
for s in range(40):
    obs, info = env.reset(seed=s)
    names=[n for n in obs.get_object_names() if n.startswith('block')]
    sh=obs.get_object_from_name('shelf'); x1=obs.get(sh,'x1'); w1=obs.get(sh,'width1')
    inside=[n for n in names if min(p[1] for p in block_corners(blk(obs,n)))>=2.625 and block_corners(blk(obs,n))[:,0].min()>=x1-1e-6 and block_corners(blk(obs,n))[:,0].max()<=x1+w1+1e-6]
    k=(len(names),len(inside),round(w1,3)); cnt[k]=cnt.get(k,0)+1
print('(n_blocks, n_inside, inner_width): count', cnt)
for oc in [1,2]:
    try:
        obs, info = env.reset(seed=0, options={'object_count':oc}); print('oc',oc,info, sorted(obs.get_object_names()))
        obs,r,term,tr,_=env.step(act()); print('term step1',term)
    except Exception as e: print('err',e)
