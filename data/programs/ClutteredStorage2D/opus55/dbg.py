import sys
from test_approach import run
import approach
seed, oc = int(sys.argv[1]), int(sys.argv[2])
print(run(seed, oc, verbose=True))
