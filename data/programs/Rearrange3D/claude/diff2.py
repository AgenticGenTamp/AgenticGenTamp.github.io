import numpy as np
from png_tool import read_png, write_png
D='/sandbox/mcp_renders/state_custom_policy_seed0_%04d.png'
a=read_png(D%35).astype(float); b=read_png(D%79).astype(float)
d=np.abs(a-b).sum(axis=2); m=d>20
ys,xs=np.nonzero(m)
import collections
print("row hist:", sorted(collections.Counter(ys//20*20).items()))
print("col hist:", sorted(collections.Counter(xs//20*20).items()))
