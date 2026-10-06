import numpy as np, zlib, base64
L=np.load('scratch/lib_all.npy').astype(np.float32)
arm=np.ascontiguousarray(L[:,:7])
s=base64.b64encode(zlib.compress(arm.tobytes(),9)).decode()
lines=[s[i:i+100] for i in range(0,len(s),100)]
with open('grasp_lib_data.py','w') as f:
    f.write('"""Precomputed PR2 left-arm configurations (float32, Nx7, zlib+base64)."""\n')
    f.write('N = %d\nARM_B64 = (\n'%len(arm))
    for l in lines: f.write('    "%s"\n'%l)
    f.write(')\n')
print(len(s))
