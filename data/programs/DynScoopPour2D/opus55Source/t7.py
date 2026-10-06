import sys
import approach
from approach import GeneratedApproach
GeneratedApproach._choose_descent=lambda self: min(1.6, 1.49 + self.off[0])
sys.argv=['x']+sys.argv[1:]
exec(open('test_approach.py').read())
