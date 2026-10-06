# Reuse twin across episodes with different seeds; compare overlap with fresh ref env.
import numpy as np, pybullet as p
from planner import TwinPlanner
from kinder.envs.kinematic3d.packing3d import ObjectCentricPacking3DEnv
tp = TwinPlanner(3); env = tp.env
ref = ObjectCentricPacking3DEnv(num_parts=3, allow_state_access=True)
rng = np.random.default_rng(0); mism = 0; nover = 0
for seed in range(8):
    ref.reset(seed=seed); st = ref.get_state(); tp.set_state(st)
    names = list(env._part_ids)
    for it in range(150):
        for nm in names:
            pos = (0.3 + rng.uniform(-0.08, 0.08), rng.uniform(-0.1, 0.1), 0.1 + rng.uniform(0, 0.05))
            orn = p.getQuaternionFromEuler(rng.uniform(-np.pi, np.pi, 3) * np.array([0.3, 0.3, 1.0]))
            for e in (env, ref):
                p.resetBasePositionAndOrientation(e._object_name_to_pybullet_id(nm), pos, orn, physicsClientId=e.physics_client_id)
        for i in range(3):
            for j in range(i + 1, 3):
                a = env._parts_overlap(names[i], names[j]); b = ref._parts_overlap(names[i], names[j])
                mism += a != b; nover += b
print("mismatch", mism, "overlaps", nover)
