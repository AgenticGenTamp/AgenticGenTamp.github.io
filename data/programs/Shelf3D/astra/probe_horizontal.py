from probe_tilt import run
import numpy as np
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(2) as ex:
 list(ex.map(run,[(x,z,-np.pi/2) for x in [.6,.7] for z in [.3,.6]]))
