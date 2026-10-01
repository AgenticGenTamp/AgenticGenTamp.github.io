from probe_shelf import run
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(3) as ex:list(ex.map(run,[(.5,.13),(.6,.13),(.7,.13),(.6,.40),(.6,.43),(.6,.70)]))
