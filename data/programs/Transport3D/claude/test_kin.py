"""Sanity checks for kin.py (Kinova Gen3 7-DOF kinematics)."""
import time

import numpy as np

import kin

np.set_printoptions(precision=4, suppress=True)


def show(name, q, **kw):
    T = kin.fk(q, **kw)
    print(f"--- {name}  (kw={kw})")
    print("  q        =", np.round(np.asarray(q, float), 4))
    print("  position =", np.round(T[:3, 3], 4))
    print("  R =\n", np.array2string(np.round(T[:3, :3], 4), prefix="       "))
    print("  quat(xyzw)=", np.round(kin.mat_to_quat(T[:3, :3]), 4))
    print("  ee_z_axis =", np.round(kin.ee_z_axis(q, **kw), 4))
    return T


print("=" * 70)
print("DH table (modified/Craig): A_i = Rx(alpha_i-1) Tx(a_i-1) Rz(th_i+off) Tz(d_i)")
print("   i   alpha      a         d        theta_offset")
for i, (al, a, d, off) in enumerate(kin.DH, 1):
    print(f"  {i}   {al:+.5f}  {a:+.4f}  {d:+.4f}   {off:+.5f}")
print("  frame7 -> tool: Rx(pi)   (+z becomes the approach axis)")
print("=" * 70)

T_ret = show("RETRACT (0,-0.35,-3.1416,-2.5,0,-0.87,1.5708)", kin.Q_RETRACT)
T_home = show("HOME    (0, 0.26, 3.14, -2.27, 0, 0.96, 1.57)", kin.Q_HOME)

print()
print("Kinova web-app reference for HOME: p ~ (0.457, 0.001, 0.433)")
print("  error vs reference:",
      np.round(T_home[:3, 3] - np.array([0.457, 0.001, 0.433]), 4))

# --- reach check -----------------------------------------------------------
print()
print("--- reach check")
zero = kin.fk(np.zeros(7))[:3, 3]
print("  fk(0) position          =", np.round(zero, 4), " |p| =", round(float(np.linalg.norm(zero)), 4))
rng = np.random.default_rng(0)
qs = rng.uniform(-np.pi, np.pi, (4000, 7))
qs[:, 1] = np.clip(qs[:, 1], -2.41, 2.41)
qs[:, 3] = np.clip(qs[:, 3], -2.66, 2.66)
qs[:, 5] = np.clip(qs[:, 5], -2.23, 2.23)
r = np.array([np.linalg.norm(kin.fk(q)[:3, 3]) for q in qs])
print(f"  max |p| over 4000 random configs = {r.max():.4f} m (Gen3 spec reach ~0.90 m)")

# --- tool_z / mount / base_rot ---------------------------------------------
print()
print("--- tool_z, mount and base_rot")
p0 = kin.fk(kin.Q_HOME)[:3, 3]
p1 = kin.fk(kin.Q_HOME, tool_z=0.12)[:3, 3]
z = kin.ee_z_axis(kin.Q_HOME)
print("  tool_z=0.12 shifts by", np.round(p1 - p0, 4), "expected", np.round(0.12 * z, 4))
p2 = kin.fk(kin.Q_HOME, base_x=1.0, base_y=-0.5, mount=(0, 0, 0.3))[:3, 3]
print("  base(1,-0.5)+mount z0.3 ->", np.round(p2, 4), " (== p0 + offset:", np.round(p0 + [1, -0.5, 0.3], 4), ")")
p3 = kin.fk(kin.Q_HOME, base_rot=np.pi / 2)[:3, 3]
print("  base_rot=90deg          ->", np.round(p3, 4), " (expect x/y swap of", np.round(p0, 4), ")")

# --- jacobian check --------------------------------------------------------
print()
print("--- jacobian")
J = kin.jacobian(kin.Q_HOME)
print("  shape", J.shape, " sigma_min=%.4f sigma_max=%.4f" % tuple(
    np.linalg.svd(J, compute_uv=False)[[-1, 0]]))
dq = np.array([0.01, -0.01, 0.005, 0.01, -0.005, 0.01, 0.0])
pred = (J @ dq)[:3]
act = kin.fk(kin.Q_HOME + dq)[:3, 3] - kin.fk(kin.Q_HOME)[:3, 3]
print("  J*dq =", np.round(pred, 5), " actual dp =", np.round(act, 5),
      " err =", float(np.round(np.linalg.norm(pred - act), 7)))

# --- IK --------------------------------------------------------------------
print()
print("--- IK: round trip from random reachable configs (pos + orientation)")
ok = 0
times = []
errs = []
rng = np.random.default_rng(7)
for _ in range(25):
    qt = kin.clamp_to_limits(kin.Q_HOME + rng.uniform(-0.7, 0.7, 7))
    T = kin.fk(qt, tool_z=0.12)
    quat = kin.mat_to_quat(T[:3, :3])
    t0 = time.perf_counter()
    q, info = kin.ik(T[:3, 3], quat, q_init=kin.Q_HOME, tool_z=0.12, return_info=True)
    times.append((time.perf_counter() - t0) * 1e3)
    Ts = kin.fk(q, tool_z=0.12)
    e = np.linalg.norm(Ts[:3, 3] - T[:3, 3])
    errs.append(e)
    ok += info["success"] and kin.in_limits(q)
print(f"  solved {ok}/25   max pos err {max(errs)*1000:.3f} mm")
print(f"  time: median {np.median(times):.1f} ms  mean {np.mean(times):.1f} ms  max {max(times):.1f} ms")

print()
print("--- IK: top-down grasps over a table workspace")
ok = 0
times = []
zerrs = []
targets = []
for x in (0.35, 0.5, 0.65):
    for y in (-0.25, 0.0, 0.25):
        for zz in (0.05, 0.25):
            targets.append((x, y, zz))
for t in targets:
    t0 = time.perf_counter()
    q, info = kin.ik_top_down(np.array(t), yaw=0.0, q_init=kin.Q_HOME,
                              tool_z=0.12, return_info=True)
    times.append((time.perf_counter() - t0) * 1e3)
    zax = kin.ee_z_axis(q, tool_z=0.12)
    zerrs.append(np.linalg.norm(zax - np.array([0, 0, -1.0])))
    ok += info["success"] and kin.in_limits(q)
print(f"  solved {ok}/{len(targets)}   max |ee_z - (0,0,-1)| = {max(zerrs):.4f}")
print(f"  time: median {np.median(times):.1f} ms  max {max(times):.1f} ms")

print()
print("--- IK: position-only (target_quat=None)")
t0 = time.perf_counter()
q, info = kin.ik([0.5, 0.1, 0.3], None, q_init=kin.Q_HOME, return_info=True)
dt = (time.perf_counter() - t0) * 1e3
print("  ", info, " p =", np.round(kin.fk(q)[:3, 3], 4), f" {dt:.1f} ms")
print("   within limits:", kin.in_limits(q))

print()
print("--- helpers")
print("  top_down_quat(0)   =", np.round(kin.top_down_quat(0.0), 4))
print("  quat->mat->quat ok :", np.allclose(
    kin.quat_to_mat(kin.top_down_quat(0.3)), kin.quat_to_mat(kin.mat_to_quat(
        kin.quat_to_mat(kin.top_down_quat(0.3))))))
