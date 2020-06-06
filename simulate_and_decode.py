#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoder testing script
"""
__author__ = "Christoph Kirst, University of California, San Francisco"
__license__ = "MIT License"

import numpy as np;
import matplotlib.pyplot as plt

import scipy as sp


#%% general parameter

#model parameter
model = dict(spike_rates = [4,7,2,4,1],  #Hz
             position_centers = [2, 5, 10, 15, 17],    #linear position bin number
             position_std = [3, 6, 4, 1, 5],
             mark_centers = [[60],[100],[200],[300],[150]],
             mark_std = [[2],[2],[2],[2],[2]]); 

#derived parameter
model['n_neurons'] = len(model['spike_rates']);


parameter = dict(time_bin = 10, # ms           # the length of a single time bin
                 position_bin = 5, # cm        # the size of a single position bin
                 positions = np.arange(0,21),  # the list of position bins (linearized)
                 
                 encoding = dict(
                   position_kernel_std = 1, 
                   mark_kernel_std = 20),
                 
                 decoding = dict(transition_std = 3, 
                                 transition_gain = 0.01,
                                 time_steps_per_bin = 1)       # number of time steps used for the decodind time bin
                )

#derived parameter
position_kernel_std = parameter['encoding']['position_kernel_std'];
position_kernel_range = 6 * position_kernel_std;
parameter['encoding']['position_kernel'] = sp.stats.norm.pdf(np.arange(0,2*position_kernel_range+1,1), position_kernel_range, position_kernel_std)


#%% Generate test trajectory 

def generate_trajectory(total_time = 20):
  # total_time in sec
  sample_rate = 1000/parameter['time_bin']
  time_steps = np.arange(total_time * sample_rate)
  times = time_steps / sample_rate;
  
  max_velocity = 15; # cm/sec
  positions_max = np.max(parameter['positions'])
  trajectory = np.array(np.round(positions_max * (np.cos(max_velocity/parameter['position_bin'] * times) + 1) / 2), dtype=int);
  
  trajectory_data = dict(positions=trajectory,
                         times = times,
                         time_steps=time_steps);
  
  return trajectory_data;

#%%
trajectory_data = generate_trajectory(20);
plt.figure(1); plt.clf();
plt.plot(trajectory_data['times'], trajectory_data['positions'])


#%% Generate spikes along the trajectory according to model

def generate_spikes(model, parameter, trajectory_data):
  """"Generates a spike train of a marked inhomogeneous Poisson process"""
  
  n_neurons = len(model['spike_rates']);
  spike_rates = model['spike_rates'];
  position_centers = model['position_centers'];
  position_std = model['position_std'];
  mark_centers = model['mark_centers'];
  mark_std = model['mark_std'];
  
  times = trajectory_data['times'];
  time_steps = trajectory_data['time_steps'];
  trajectory = trajectory_data['positions'];
  
  unit_data = {}
  for n in range(n_neurons):
    #generate spikes by approximating inhomogenous Poission process with Bernoulli process
    pos_rv = sp.stats.norm(loc=position_centers[n], scale=position_std[n])
    spike_rate = pos_rv.pdf(trajectory) / pos_rv.pdf(position_centers[n]);
    spike_rate = spike_rate / np.sum(spike_rate) * len(spike_rate) * np.max(times)/np.max(time_steps) * spike_rates[n]
    spike_rate[spike_rate > 1] = 1;
    spikes = sp.stats.bernoulli(p=spike_rate).rvs() > 0;
    n_spikes = np.sum(spikes);
    
    #generate marks - assumes mark probability is uniformly distributed over position.
    mark_rv = sp.stats.multivariate_normal(mean=mark_centers[n], cov=np.diag(mark_std[n]));
    marks = mark_rv.rvs(n_spikes);
    if marks.ndim == 1:
      marks = marks[:,None];
    
    #genereate unit spike data
    unit_data[n] = dict(positions = trajectory[spikes], 
                        marks=marks, 
                        time_steps = time_steps[spikes],
                        times = times[spikes]);
    
  #generate full data 
  data = dict();
  data['unit'] = np.hstack([np.ones(len(unit_data[n]['times'])) * n for n in range(n_neurons)]);
  for q in ['positions', 'marks', 'times', 'time_steps']:
    if q == 'marks':
      data[q] = np.vstack([unit_data[n][q] for n in range(n_neurons)]);
    else:
      data[q] = np.hstack([unit_data[n][q] for n in range(n_neurons)]);
  sort = np.argsort(data['time_steps']);
  for q in data.keys():
    data[q] = data[q][sort];
  return data;
  
#%% 

spikes = generate_spikes(model, parameter, trajectory_data)

plt.figure(2); plt.clf();
plt.plot(spikes['times'], np.array(model['position_centers'])[np.asarray(spikes['unit'], dtype=int)], '|')
plt.plot(trajectory_data['times'], trajectory_data['positions'])

#%% check spike rate statistics

spike_rate_measures = [];
total_time = trajectory_data['times'][-1];
for i in range(500):
  print(i);
  spikes = generate_spikes(model, parameter, trajectory_data)
  spike_rate_measures.append([(np.sum(spikes['unit']==i)/total_time) for i in range(len(model['spike_rates']))]);
spike_rate_measures = np.array(spike_rate_measures);

plt.figure(5); plt.clf();
for i in range(len(model['spike_rates'])):
  print(i)
  plt.hist(spike_rate_measures[:,i], label='Unit %d' % i)
plt.xlabel('spike rate r [Hz]')
plt.ylabel('p(r)')
plt.legend()


#%% encoding data to build kernel density econding model

encode_trajectory_data = generate_trajectory(total_time=19);
encode_data = generate_spikes(model, parameter, encode_trajectory_data)

plt.figure(2); plt.clf();
plt.plot(encode_data['times'], np.array(model['position_centers'])[np.asarray(encode_data['unit'], dtype=int)], '|')
plt.plot(encode_trajectory_data['times'], encode_trajectory_data['positions'])


#%% decoding data
decode_total_time = 10;
decode_trajectory_data = generate_trajectory(total_time=decode_total_time);
decode_data = generate_spikes(model, parameter, decode_trajectory_data)

plt.figure(3); plt.clf();
plt.plot(decode_data['times'], np.array(model['position_centers'])[np.asarray(decode_data['unit'], dtype=int)], '|')
plt.plot(decode_trajectory_data['times'], decode_trajectory_data['positions'])

#%% decode

import decoder as dec

likelihoods, posteriors= dec.decode(encode_data, encode_trajectory_data, decode_data, decode_total_time, parameter);

#%%

plt.figure(42); plt.clf();
plt.imshow(posteriors.T, interpolation=None, aspect='auto', extent=(0, decode_total_time, *parameter['positions'][[0,-1]]), origin='lower')
plt.plot(decode_trajectory_data['times'], decode_trajectory_data['positions'], c='red', linewidth=2)

#max posterior
position_hat = np.argmax(posteriors, axis=1)
plt.plot(decode_trajectory_data['times'], position_hat, c='white', linewidth=2)



