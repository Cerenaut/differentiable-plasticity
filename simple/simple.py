# Differentiable plasticity: simple binary pattern memorization and reconstruction.
#
# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This program is meant as a simple instructional example for differentiable plasticity. It is fully functional but not very flexible.

# Usage: python simple.py [rngseed], where rngseed is an optional parameter specifying the seed of the random number generator. 
# To use it on a GPU or CPU, toggle comments on the 'ttype' declaration below.



import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
import random
import sys
import pickle as pickle
import pdb
import time
import logging


PATTERNSIZE = 1000
NBNEUR = PATTERNSIZE+1  # NbNeur = Pattern Size + 1 "bias", fixed-output neuron (bias neuron not needed for this task, but included for completeness)
#ETA = .01               # The "learning rate" of plastic connections - we actually learn it
ADAMLEARNINGRATE = 3e-4  # The learning rate of the Adam optimizer
RNGSEED = 0             # Initial random seed - can be modified by passing a number as command-line argument

# Note that these patterns are likely not optimal
PROBADEGRADE = .5       # Proportion of bits to zero out in the target pattern at test time
NBPATTERNS = 5          # The number of patterns to learn in each episode
NBPRESCYCLES = 2        # Number of times each pattern is to be presented
PRESTIME = 6            # Number of time steps for each presentation
PRESTIMETEST = 6        # Same thing but for the final test pattern
INTERPRESDELAY = 0 # 4      # Duration of zero-input interval between presentations
NBSTEPS = NBPRESCYCLES * ((PRESTIME + INTERPRESDELAY) * NBPATTERNS) + PRESTIMETEST  # Total number of steps per episode


if len(sys.argv) == 2:
    RNGSEED = int(sys.argv[1])
    print("Setting RNGSEED to "+str(RNGSEED))
np.set_printoptions(precision=3)
np.random.seed(RNGSEED); random.seed(RNGSEED); torch.manual_seed(RNGSEED)

ITNS = 2000
BPIT = False
LOAD_PARAMS_FROM_DISK = True
SPARSITY = 0.5    # fractional sparsity e.g. 0.5 = 0.5 active,   0.2 = 0.8 active

MIN_BIT_VALUE = -1
DEGRADE_VALUE = 0   # because the input is amplified, anything other than 0 has a similar effect i.e. still ends up being greater than 1 or less than -1

print_every = 10

dbug_pickle = False
dbug_bp = False

# ttype = torch.FloatTensor;         # For CPU
ttype = torch.cuda.FloatTensor;     # For GPU


def zero_if_less_than(x, eps):
    """Return 0 if x<eps, otherwise return x"""
    if x < eps:
        return 0
    else:
        return x


# Generate the full list of inputs for an episode.
# The inputs are returned as a PyTorch tensor of shape NbSteps x 1 x NbNeur
def generateInputsAndTarget():
    inputT = np.zeros((NBSTEPS, 1, NBNEUR)) #inputTensor, initially in numpy format...

    # Create the random patterns to be memorized in an episode
    length_sparse = int(PATTERNSIZE * SPARSITY)
    seedp = np.ones(PATTERNSIZE); seedp[:length_sparse] = MIN_BIT_VALUE
    patterns=[]
    for nump in range(NBPATTERNS):
        p = np.random.permutation(seedp)
        patterns.append(p)

    # Now 'patterns' contains the NBPATTERNS patterns to be memorized in this episode - in numpy format
    # Choosing the test pattern, partially zero'ed out, that the network will have to complete
    testpattern = random.choice(patterns).copy()
    preservedbits = np.ones(PATTERNSIZE); preservedbits[:int(PROBADEGRADE * PATTERNSIZE)] = DEGRADE_VALUE; np.random.shuffle(preservedbits)
    degradedtestpattern = testpattern * preservedbits

    logging.debug("test pattern     = ", testpattern)
    logging.debug("degraded pattern = ", degradedtestpattern)

    # Inserting the inputs in the input tensor at the proper places
    for nc in range(NBPRESCYCLES):
        np.random.shuffle(patterns)
        for ii in range(NBPATTERNS):
            for nn in range(PRESTIME):
                numi =nc * (NBPATTERNS * (PRESTIME+INTERPRESDELAY)) + ii * (PRESTIME+INTERPRESDELAY) + nn
                inputT[numi][0][:PATTERNSIZE] = patterns[ii][:]

    # Inserting the degraded pattern
    for nn in range(PRESTIMETEST):
        logging.debug("insert degraded pattern at: [{0},{1},:{2}]".format(-PRESTIMETEST + nn, 0, PATTERNSIZE))
        inputT[-PRESTIMETEST + nn][0][:PATTERNSIZE] = degradedtestpattern[:]

    for nn in range(NBSTEPS):
        inputT[nn][0][-1] = 1.0  # Bias neuron.
        inputT[nn] *= 20.0       # Strengthen inputs
    inputT = torch.from_numpy(inputT).type(ttype)  # Convert from numpy to Tensor
    target = torch.from_numpy(testpattern).type(ttype)

    logging.debug("shape of inputT: ", np.shape(inputT))

    return inputT, target


class NETWORK(nn.Module):
    def __init__(self):
        super(NETWORK, self).__init__()

        if BPIT:
            self.w_default = 0.01
            self.alpha_default = 0.01
            self.eta_default = 0.01
        else:
            self.w_default = 0
            self.alpha_default = 0.02
            self.eta_default = 0.01

        # Notice that the vectors are row vectors, and the matrices are transposed wrt the usual order, following
        # apparent pytorch conventions Each *column* of w targets a single output neuron
        self.w = Variable(self.w_default * torch.randn(NBNEUR, NBNEUR).type(ttype), requires_grad=True)   # The matrix of fixed (baseline) weights
        self.alpha = Variable(self.alpha_default * torch.randn(NBNEUR, NBNEUR).type(ttype), requires_grad=True)  # The matrix of plasticity coefficients
        self.eta = Variable(self.eta_default * torch.ones(1).type(ttype), requires_grad=True)  # The weight decay term / "learning rate" of plasticity - trainable, but shared across all connections

    def forward(self, input, yin, hebb):
        # Run the network for one timestep
        yout = F.tanh( yin.mm(self.w + torch.mul(self.alpha, hebb)) + input )
        hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0] # bmm here is used to implement an outer product between yin and yout, with the help of unsqueeze (i.e. added empty dimensions)
        return yout, hebb

    def initialZeroState(self):
        # Return an initialized, all-zero hidden state
        return Variable(torch.zeros(1, NBNEUR).type(ttype))

    def initialZeroHebb(self):
        # Return an initialized, all-zero Hebbian trace
        return Variable(torch.zeros(NBNEUR, NBNEUR).type(ttype))


net = NETWORK()
optimizer = torch.optim.Adam([net.w, net.alpha, net.eta], lr=ADAMLEARNINGRATE)
total_loss = 0.0; all_losses = []
nowtime = time.time()


# override defaults if loading from disk
if LOAD_PARAMS_FROM_DISK:
  fn = './results/output_simple_base.dat'
  with open(fn, 'rb') as fo:
    myw = pickle.load(fo)
    myalpha = pickle.load(fo)
    myy = pickle.load(fo)
    myall_losses = pickle.load(fo)
    myeta = pickle.load(fo)

    net.w.data = torch.from_numpy(myw).type(ttype)
    net.alpha.data = torch.from_numpy(myalpha).type(ttype)
    net.eta.data = torch.from_numpy(myeta).type(ttype)

    if dbug_pickle:
        print("loading w: ", myw)
        print("loading alpha: ", myalpha)
        print("loading eta: ", myeta)

    for numiter in range(ITNS):
        generateInputsAndTarget()

init_net = False
for numiter in range(ITNS):

    print("iter: ", numiter)

    # Initialize network for each episode
    if not init_net:
        y = net.initialZeroState()
        hebb = net.initialZeroHebb()
        init_net = True
    if BPIT:
        optimizer.zero_grad()

    # Generate the inputs and target pattern for this episode
    inputs, target = generateInputsAndTarget()

    # Run the episode!
    for numstep in range(NBSTEPS):
        y, hebb = net(Variable(inputs[numstep], requires_grad=False), y, hebb)

    # Compute loss for this episode (last step only)
    loss = (y[0][:PATTERNSIZE] - Variable(target, requires_grad=False)).pow(2).sum()

    dbug_pickle = False
    dbug_bp = False
    if dbug_bp:
        print("NBSTEPS = ", NBSTEPS)
        print("Expected bp functions = ", NBSTEPS * 3 + 2)

        curr_fn = loss.grad_fn
        print(curr_fn)
        i = 0
        while curr_fn is not None:
            curr_fn = curr_fn.next_functions[0][0]
            print(i, curr_fn)
            i += 1

    if BPIT:
        # Apply backpropagation to adapt basic weights and plasticity coefficients
        loss.backward()
        optimizer.step()

    # That's it for the actual algorithm!
    # Print statistics, save files
    #lossnum = loss.data[0]   # Saved loss is the actual learning loss (MSE)
    to = target.cpu().numpy(); yo = y.data.cpu().numpy()[0][:PATTERNSIZE]; z = (np.sign(yo) != np.sign(to)); lossnum = np.mean(z)  # Saved loss is the error rate

    total_loss  += lossnum
    if (numiter+1) % print_every == 0:
        print((numiter, "===="))
        print("T", target.cpu().numpy()[-10:])   # Target pattern to be reconstructed
        print("I", inputs.cpu().numpy()[numstep][0][-11:])  # Last input contains the degraded pattern fed to the network at test time (last num is bias neuron)
        print("Y", y.data.cpu().numpy()[0][-11:])   # Final output of the network

        diff = y.data.cpu().numpy()[0][:PATTERNSIZE] - target.cpu().numpy()[:]
        vfunc = np.vectorize(zero_if_less_than)
        vfunc(diff, 0.01)
        print("D", diff[-10:])

        previoustime = nowtime
        nowtime = time.time()
        print("Time spent on last", print_every, "iters: ", nowtime - previoustime)
        total_loss /= print_every
        all_losses.append(total_loss)
        print("Mean loss over last", print_every, "iters:", total_loss)
        print("")
        with open('./results/output_simple_'+str(RNGSEED)+'.dat', 'wb') as fo:
            pickle.dump(net.w.data.cpu().numpy(), fo)
            pickle.dump(net.alpha.data.cpu().numpy(), fo)
            pickle.dump(y.data.cpu().numpy(), fo)  # The final y for this episode
            pickle.dump(all_losses, fo)
            pickle.dump(net.eta.data.cpu().numpy(), fo)

            if dbug_pickle:
                print("save w: ", net.w.data)
                print("save alpha: ", net.alpha.data)
                print("save eta: ", net.eta.data)

        with open('./results/loss_simple_'+str(RNGSEED)+'.txt', 'w') as fo:
            for item in all_losses:
                fo.write("%s\n" % item)
        total_loss = 0



