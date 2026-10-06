from plow_search import *
Rd = np.column_stack([[0,1,0],[1,0,0],[0,0,-1]]).astype(float)
for r in [0.35,0.45,0.55,0.65]:
    for z in [0.03,0.05]:
        q,ep,er=ikn(np.array([r,0,z]),Rd,q_init,q_init,iters=500)
        pr=profile(q)
        lim = [(-2.24,2.24),(-2.57,2.57),(-2.09,2.09)]
        print(r,z,'err %.1e %.1e'%(ep,er),'maxdq %.2f'%np.max(np.abs(q-q_init)), np.round(q,2).tolist(), 'prof', np.round(pr,3) if pr else None)
