import numpy as np
from png_tool import read_png, write_png
a=np.load('a.npy'); b=np.load('b.npy')
y0,y1,x0,x1 = 170, 290, 265, 380   # gripper bbox (295,191)-(349,264) + generous margin
S=8
def crop(img):
    c = img[y0:y1, x0:x1]
    return np.repeat(np.repeat(c, S, axis=0), S, axis=1)
ca, cb = crop(a), crop(b)
gap = np.full((ca.shape[0], 24, 3), 255, np.uint8)
out = np.concatenate([ca, gap, cb], axis=1)
# red border on each panel for clarity
out[:3,:]=[255,0,0]; out[-3:,:]=[255,0,0]
write_png('mcp_renders/zoom01.png', out)
print(out.shape, 'left=grip0.0  right=grip1.0')
print('crops equal:', np.array_equal(ca, cb))
