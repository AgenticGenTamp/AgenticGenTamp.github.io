import subprocess, numpy as np
W,H=640,360
def load(p):
    raw=subprocess.run(['convert',p,'-depth','8','rgb:-'],capture_output=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(H,W,3).astype(int)
im=load('mcp_renders/state_custom_policy_seed0_0009_1.png')
R,G,B=im[...,0],im[...,1],im[...,2]
neutral=(np.abs(R-G)<=6)&(np.abs(G-B)<=6)&(np.abs(R-B)<=6)
purple=(R>35)&(G<45)&(B>35)&(np.abs(R-B)<25)&(R>2*G+10)
def cls(y,x):
    if purple[y,x]: return 'P'
    v=R[y,x]
    if neutral[y,x]:
        if v<=45: return '#'   # gripper black
        if v<=150: return '*'  # mid gray (finger/link)
        if v<=215: return 'o'  # light gray fingertip / shaded plate
        return 'W'             # white plate
    return '.'                 # table/wall
print('    '+''.join(str((250+i)//10%10) for i in range(65)))
print('    '+''.join(str((250+i)%10) for i in range(65)))
for y in range(95,146):
    print(f'{y:3d} '+''.join(cls(y,x) for x in range(250,315)))
