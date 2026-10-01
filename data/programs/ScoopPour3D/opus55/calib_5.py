import numpy as np, sys
exec(open('calib_4.py').read().split("for g in")[0])
# wall top heights (closed)
print('long wall top (0.5,-0.05) closed:', touch((0.5,-0.05),1.0,z0=0.62,zmin=0.49))
print('short wall top (0.275,-0.2) closed:', touch((0.275,-0.2),1.0,z0=0.62,zmin=0.49))
print('bin', r.P('bin_yellow_0'))
for d in np.arange(-0.07,0.071,0.01):
    l,b=touch((0.5,-0.05+d),0.0,z0=0.6,zmin=0.50)
    print('ysweep d=%.3f'%d, 'last free', None if l is None else round(l,4))
for d in np.arange(-0.07,0.071,0.01):
    l,b=touch((0.275+d,-0.2),0.0,z0=0.6,zmin=0.50)
    print('xsweep d=%.3f'%d, 'last free', None if l is None else round(l,4))
print('bin', r.P('bin_yellow_0'))
