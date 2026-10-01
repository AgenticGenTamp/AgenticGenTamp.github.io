import sys, json
import approach
approach.GeneratedApproach.CLOSE_LONG = sys.argv[1]=='1'
approach.GeneratedApproach.SLOTS_OVERRIDE = json.loads(sys.argv[2])
import test_approach as T
c=int(sys.argv[3])
for sd in map(int,sys.argv[4:]): print(sd, T.run(sd,count=c))
