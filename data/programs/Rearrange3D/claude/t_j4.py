import numpy as np
from env_client import make_env
env=make_env()
# refine limits for joints 2,4,6 with fine steps near boundary
for j,lo,hi in [(1,-2.2343,1.2778),(3,-2.5833,2.5833),(5,-2.1091,2.1091)]:
    print(f"joint{j+1} observed clamp: [{lo}, {hi}]")
env.close()
