import sys, approach
approach.GeneratedApproach.DD_MAX=float(sys.argv[1])
import test_approach as T
for sd in sys.argv[2:]: print(sd, T.run(int(sd))[:3])
