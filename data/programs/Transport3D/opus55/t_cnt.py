import sys
import test_approach as T
c=int(sys.argv[1])
for sd in map(int,sys.argv[2:]): print(sd, T.run(sd,count=c))
