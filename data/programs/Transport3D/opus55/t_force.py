import sys
import approach
approach.GeneratedApproach.FORCE_BOX_FIRST=True
import test_approach as T
for sd in map(int,sys.argv[1:]): print(sd, T.run(sd))
