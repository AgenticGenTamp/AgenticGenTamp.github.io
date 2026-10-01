from geo_lib import *
P=Prober(7,'part0')
print('xmax',bisect(P,lambda t:(t,0.0),0.3,0.4))
print('xmin',bisect(P,lambda t:(t,0.0),0.3,0.2))
print('ymax',bisect(P,lambda t:(0.3,t),0.0,0.15))
print('ymin',bisect(P,lambda t:(0.3,t),0.0,-0.15))
print('steps',P.nsteps)
