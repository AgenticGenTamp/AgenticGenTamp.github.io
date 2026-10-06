import sys
from test_approach import run
# run a few episodes sequentially in one process, then the target
for s,c in [(19,30),(20,30),(21,30)]:
    print(s,c,run(s,c),flush=True)
