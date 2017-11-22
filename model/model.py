#! /usr/bin/env python

import numpy as np
import tools.bit_opt as bt
import tools.fxd as fxd
import pickle

def mult_profile(bit_width, area, time, k):
    cycles = np.arange(((bit_width + 2)/2)*k+1, (((bit_width + 2)/2)), -1*(k-1))
    return area*time*cycles

def mult_perf(vals, at, bit_width=0):
   
    if bit_width == 0: bit_width = (len(at)-2)*2

    hist = np.zeros(len(at), np.int32)
    
    for val in vals:
        fl = fxd.calculate_fl_np(val, bit_width)
        val_fxd = fxd.to_fixed_np(val, fl)
        x_dict = bt.get_skip_val_v(val_fxd, (len(at)-2)*2)

        for i in range(0,len(at)):
            hist[i] += (x_dict == i).sum()

    dist = hist/hist.sum()
    avg_at = (dist*at).sum()
    print(dist)
    print(avg_at, end="")
    print(" Avg Latency: ", end="")
    print(((len(at)-2)*2)+2 - (np.abs(at - avg_at)).argmin())


at = mult_profile(16, 30, 1.52, 2)
#at = mult_profile(32, 55, 1.76, 2)
#at = mult_profile(64, 105, 1.83, 2)
print(at)

num = 10000
sp = 0

# Distribution Performance for Gaussian
print("Gaussian: ", end="")
mu, sigma = 0, 0.1
s = np.random.normal(mu, sigma, int(num*(1-sp)))
s = np.append(s, np.zeros(int(num*sp)))
mult_perf(s, at)
mult_perf(s, at, 8)

# Distribution Performance for Uniform 
print("Uniform: ", end="")
low, high = -1, 1
s = np.random.uniform(low, high, int(num*(1-sp)))
s = np.append(s, np.zeros(int(num*sp)))
mult_perf(s, at)

# Distribution Performance for AlexNet
print("AlexNet: ", end="")
file_object = open('alexnet.p', 'rb')
weights = pickle.load(file_object)
mult_perf(weights, at)

# Distribution Performance for LeNet
print("LeNet: ", end="")
file_object = open('lenet.p', 'rb')
weights = pickle.load(file_object)
mult_perf(weights, at)

# Distribution Performance for LeNet
print("LeNet(Sparse): ", end="")
file_object = open('lenet_sparse.p', 'rb')
weights = pickle.load(file_object)
mult_perf(weights, at)
