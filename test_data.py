import h5py
import numpy as np
from graphics import *

f = h5py.File('data.h5','r')

pixels = []
vys = f['vys']
jump = f['jump']

for key in f.keys():
    if 'pixel' in key:
        pixels.append(np.array(f[key]))

f.close()

pixels = np.concatenate(pixels,axis=0)

print(pixels.shape)
