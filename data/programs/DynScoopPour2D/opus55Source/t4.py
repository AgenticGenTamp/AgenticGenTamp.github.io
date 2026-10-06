from common import *
env=make_env(); xs=[]
for sd in range(200):
    s,_=env.reset(seed=sd,options={'object_count':2}); xs.append(hook(s)['x'])
xs=np.array(xs); print(np.percentile(xs,[0,25,50,75,100]), (xs>3.41).mean(), (xs>3.38).mean())
