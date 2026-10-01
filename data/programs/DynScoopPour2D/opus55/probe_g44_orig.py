import sys; sys.argv=['x','0.2,0.2,0.25,0.08']
exec(open('probe_wall_10.py').read().split('cfg=')[0])
p=P(44); ok=grasp(p,0.2,0.2,0.25,0.08); print('orig',ok,p.r(),p.h())
