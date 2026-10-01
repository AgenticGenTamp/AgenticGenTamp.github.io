import time
import numpy as np
from kinova import fk, ik
rng = np.random.default_rng(1)
start = time.monotonic()
errors = []
for _ in range(30):
    q = rng.uniform(-2., 2., 7)
    target = fk(q)
    fit = ik(target[:3,3], target[:3,:3], q+rng.normal(0., .2, 7))
    got = fk(fit)
    errors.append(np.linalg.norm(got-target))
print('max transform error', max(errors), 'mean solve seconds', (time.monotonic()-start)/len(errors))
assert max(errors) < 1e-4
