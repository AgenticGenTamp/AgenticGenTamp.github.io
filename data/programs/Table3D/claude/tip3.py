import subprocess, numpy as np
def load(path):
    out = subprocess.run(['convert', path, '-depth','8','rgb:-'], capture_output=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(360,640,3).astype(int)
img=load('/sandbox/mcp_renders/state_custom_policy_seed0_0011.png')
base=load('/sandbox/mcp_renders/state_custom_policy_seed0_0000.png')
sub=img[0:80,200:300]
lum=sub.mean(2)
h,_=np.histogram(lum,bins=[0,10,20,30,40,50,60,70,90,120,160,200,256])
print('lum hist bins 0,10,...:',h)
# base wall lum in same region
print('base lum sample rows:', base[10:60:10,205:300:20].mean(2).round(1).tolist())
