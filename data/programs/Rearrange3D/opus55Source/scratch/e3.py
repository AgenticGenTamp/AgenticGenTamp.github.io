import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *; from h import _quat_yaw, _wrap
seed=int(sys.argv[1]); yaw=float(sys.argv[2]); pz=float(sys.argv[3]); mode=sys.argv[4]
h=H(seed); h.step(60); dp=h.drink()
h.setbase([-0.30, dp[1], 0]); h.move([dp[0],dp[1],0.60], yaw, 0.03); dp=h.drink()
h.move([dp[0],dp[1],pz], yaw, 0.01); h.close(25); h.rep("closed")
zmax=0
if mode=="jerk":
    h.q=h.ap.ik(h.base,[dp[0],dp[1],0.75],yaw,h.q)
    for t in range(60): h.step(); zmax=max(zmax,h.drink()[2])
elif mode=="wiggle":
    for k in range(40):
        off=0.01*np.array([np.cos(k*1.3),np.sin(k*1.3)])
        h.move([dp[0]+off[0],dp[1]+off[1],pz+0.04], yaw+0.15*np.sin(k*0.9), 0.02, settle=2); zmax=max(zmax,h.drink()[2])
elif mode=="tilt":   # lift while rolling gripper? (only yaw available in ik) -> twist
    for k in range(20):
        h.move([dp[0],dp[1],pz+0.03], yaw+(0.4 if k%2 else -0.4), 0.02, settle=2); zmax=max(zmax,h.drink()[2])
h.rep("after"); print("zmax",round(zmax,3))
h.move([dp[0],dp[1],0.65], yaw, 0.01); h.rep("lifted")
