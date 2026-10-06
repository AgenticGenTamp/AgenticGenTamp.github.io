import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *; from h import _quat_yaw, _wrap
seed=int(sys.argv[1]); yaw=float(sys.argv[2]); pz=float(sys.argv[3]) if len(sys.argv)>3 else 0.488
h=H(seed); h.step(60); dp=h.drink()
h.setbase([-0.30, dp[1], 0]); 
h.move([dp[0],dp[1],0.60], yaw, 0.03); dp=h.drink(); h.rep("pre")
h.move([dp[0],dp[1],0.60], yaw, 0.01);
h.move([dp[0],dp[1],pz], yaw, 0.01); h.rep("desc")
h.close(); h.rep("closed")
for xt in np.arange(dp[0]-0.03, 0.0, -0.03):
    h.move([xt,dp[1],pz], yaw, 0.003, settle=5); h.rep(f"pull{xt:.2f}")
h.move([0.0,dp[1],0.60], yaw, 0.01); h.rep("lift")
