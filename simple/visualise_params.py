
import torch
import numpy as np
import pickle

import matplotlib.pyplot as plt
plt.ion()


suffix='images_nbpatterns_3_nbprescycles_3_prestime_20_prestimetest_3_interpresdelay_2_patternsize_1024_nbiter_100000_probadegrade_0.5_lr_0.0001_homogenous_20_rngseed_0'
fn = './results_'+suffix+'.dat'

fn = '../simple/output_simple_0_default.dat'
# fn = '../simple/output_simple_0.dat'
with open(fn, 'rb') as fo:
    myw = pickle.load(fo)
    myalpha = pickle.load(fo)
    myy = pickle.load(fo)
    myall_losses = pickle.load(fo)

# ttype = torch.cuda.FloatTensor # Must match the one in pics_eta.py
ttype = torch.FloatTensor # Must match the one in pics_eta.py


wsize = np.shape(myw)

print("Stats of weights for each node. [Min, Max, Mean, StdDev]")
num_nodes = wsize[0]

print("------- W -------")
for i in range(num_nodes):
    print("{0:5d} - [{1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}]".format(i,
          np.amin(myw[i, :]),
          np.amax(myw[i, :]),
          np.mean(myw[i, :]),
          np.std(myw[i, :])))

print("------- alpha -------")
for i in range(num_nodes):
    print("{0:5d} - [{1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}]".format(i,
          np.amin(myalpha[i, :]),
          np.amax(myalpha[i, :]),
          np.mean(myalpha[i, :]),
          np.std(myalpha[i, :])))

print("w - [{1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}]".format(i,
          np.amin(myw),
          np.amax(myw),
          np.mean(myw),
          np.std(myw)))

print("alpha - [{1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}]".format(i,
          np.amin(myalpha),
          np.amax(myalpha),
          np.mean(myalpha),
          np.std(myalpha)))

plt.figure()
plt.subplot(1, 2, 1)  # row, col, index
plt.imshow(myw, cmap='gray', vmin=-1.0, vmax=1.0)

plt.subplot(1, 2, 2)
plt.imshow(myalpha, cmap='gray', vmin=-1.0, vmax=1.0)

plt.show(block=True)
