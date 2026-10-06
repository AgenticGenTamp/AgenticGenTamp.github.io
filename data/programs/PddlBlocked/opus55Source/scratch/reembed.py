import numpy as np, zlib, base64, glob
L=np.load('scratch/lib_all.npy')[:,:7]
H=[np.load(f) for f in sorted(glob.glob('scratch/libhard_*.npy'))]
H=[h for h in H if len(h)]
A=np.concatenate([L]+H).astype(np.float32)
print(A.shape)
np.save('scratch/arm_all.npy', A)
s=base64.b64encode(zlib.compress(np.ascontiguousarray(A).tobytes(),9)).decode()
src=open('approach.py').read()
head=src[:src.index("\n\n# ---------------------------------------------------------------- data")]
out=head+'\n\n# ---------------------------------------------------------------- data\n'
out+='_LIB_N = %d\nARM_B64 = (\n'%len(A)
for i in range(0,len(s),100): out+='    "%s"\n'%s[i:i+100]
out+=')\n'
open('approach.py','w').write(out)
