import numpy as np
from png_tool import read_png, write_png
a = read_png('mcp_renders/state_custom_grip0.png')
b = read_png('mcp_renders/state_custom_grip1.png')
print('shapes', a.shape, b.shape)
d = (a.astype(int) - b.astype(int))
m = np.abs(d).sum(axis=2) > 0
print('differing pixel count:', int(m.sum()))
print('max abs channel diff:', int(np.abs(d).max()))
if m.any():
    ys, xs = np.nonzero(m)
    print('bbox y:', ys.min(), ys.max(), 'x:', xs.min(), xs.max())
else:
    print('bbox: NONE (images pixel-identical)')
np.save('a.npy', a); np.save('b.npy', b)
