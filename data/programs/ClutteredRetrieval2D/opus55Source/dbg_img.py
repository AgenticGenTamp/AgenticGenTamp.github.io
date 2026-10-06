exec(open('dbg2.py').read().split('rng=np.random')[0])
import imageio
img=env.render(); imageio.imwrite('trap.png', img); print(img.shape)
