#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Place En-Decoder

A simple implementation of the place decoder based on Deng et al. 
"""
__author__ = "Christoph Kirst, University of California, San Francisco"
__license__ = "MIT License"


import numpy as np
import scipy as sp
import scipy.stats

import matplotlib.pyplot as plt

def position_count(data, parameter, convolve = True, epsilon = 1e-10):
  """Calculates the number of visits at each positions of a trajectory. """
  
  spike_positions = data['positions'];
  
  bin_edges = np.hstack([parameter['positions'],[parameter['positions'][-1]+1]]);
  counts, _ = np.histogram(spike_positions, bins=bin_edges)
  
  if convolve:
    counts = np.convolve(counts, parameter['encoding']['position_kernel'], mode='same')
  
  #make sure non-zero
  counts += epsilon;
  
  return counts


def occupancy(data, parameter, convolve = True, epsilon = 1e-10):
  """Calculates a smoothed version of the distribution of positions of a trajectory.
  
  Note
  ----
  This calculates the denominator of equation (14) in Deng et al.
  """
  spike_positions = data['positions'];
  
  bin_edges = np.hstack([parameter['positions'],[parameter['positions'][-1]+1]]);
  occupancy, _ = np.histogram(spike_positions, bins=bin_edges, density=True)
  
  if convolve:
    occupancy = np.convolve(occupancy, parameter['encoding']['position_kernel'], mode='same')
  
  occupancy += epsilon
  
  return occupancy


def spike_count(spike_data, parameter, convolve = True):
  """Calculates the spike rate at each position from given spike data"""   
  
  positions = spike_data['positions'];
  bin_edges = np.hstack([parameter['positions'],[parameter['positions'][-1]+1]]);
  
  spike_counts, _ = np.histogram(positions, bins=bin_edges)
  print(spike_counts.shape)
  #smooth by position kernel
  if convolve:
    spike_counts = np.convolve(spike_counts, parameter['encoding']['position_kernel'], mode='same')
  
  #delta = np.diff(parameter['positions'][:2])[0]
  #spike_counts = spike_counts / (spike_counts.sum() * delta)
  
  return spike_counts


def normal_pdf_int_lookup(x, mean, std):
    max_amp = 3000
    norm_dist = sp.stats.norm.pdf(x=np.arange(-max_amp,max_amp), loc=0, scale=std)
    return norm_dist[np.asarray(x-mean+max_amp, dtype=int)]



def intensity(encode_data, encode_trajectory_data, decode_data, parameter, normalize = True):
  """Calculates the intensity function for encoding.
  
  Returns
  -------
  l : array
    Returns the values of the intensity function at the observed 
    decoding spike marks and the positions (d,x)
    
  Note
  ----
  Calculate eq. (14) in Deng et al. for the observed marker values in the 
  spikes to decode from.
  """

  ## calculate eq. (14) in Deng et al.
  
  # location 
  positions = parameter['positions'];
  encode_spike_positions = encode_data['positions'];
  position_kernel_std = parameter['encoding']['position_kernel_std'];

  p = sp.stats.norm.pdf(np.expand_dims(positions, 0),
                        np.expand_dims(encode_spike_positions, 1),
                        position_kernel_std);

  # mark space
  encode_spike_marks = encode_data['marks'];
  decode_spike_marks = decode_data['marks'];
  n_encode_spikes = len(encode_spike_marks);
  mark_kernel_std = parameter['encoding']['mark_kernel_std'];
  m = normal_pdf_int_lookup(np.expand_dims(encode_spike_marks, 1),
                            np.expand_dims(decode_spike_marks, 0),
                            mark_kernel_std);
  m = np.prod(m, axis=2);
 
  o = occupancy(encode_trajectory_data, parameter);
  
  # lambda(x,m)
  print(m.shape, p.shape )
  l =  np.matmul(m.T, p) / o
  l = l / n_encode_spikes;
  
  # normalize lambda(x,m) -> p(x|m)
  if normalize:
    l_sum = np.nansum(l, axis=1)
    l_sum_zero = l_sum == 0
    l[l_sum_zero, :] = 1;
    l_sum[l_sum_zero] = len(positions);
    l = l / l_sum[:, np.newaxis]
        
  return l;


def gaussian_transition_matrix(parameter):      
    """Calculate a Guassian point process transition matrix."""
     
    positions = parameter['positions']
    n_positions = len(positions);
    transition_mat = np.ones([n_positions, n_positions])
    sigma = parameter['decoding']['transition_std'];
    
    for i in range(n_positions):
      transition_mat[i, :] = np.exp(-np.power(positions - positions[i], 2.) / (2 * np.power(sigma, 2.)))
    
    # uniform offset
    uniform_gain = parameter['decoding']['transition_gain'];
    uniform_dist = np.ones(transition_mat.shape)

    # normalize transition matrix
    transition_mat = transition_mat/(transition_mat.sum(axis=0)[None, :])
    transition_mat[np.isnan(transition_mat)] = 0

    # normalize uniform offset
    uniform_dist = uniform_dist/(uniform_dist.sum(axis=0)[None, :])
    uniform_dist[np.isnan(uniform_dist)] = 0

    # apply uniform offset
    transition_mat = transition_mat * (1 - uniform_gain) + uniform_dist * uniform_gain

    return transition_mat 
     
 
def decode(encode_data, encode_trajectory_data, decode_data, decode_total_time, parameter, verbose = True):
  """Decode the position from observed spikes given a set of encoding spikes."""
   
  time_steps_per_bin = parameter['decoding']['time_steps_per_bin'];
  time_delta =  parameter['time_bin'];

  #propbability of not firing a spike at location
  spike_intensity = spike_count(encode_data, parameter) / position_count(encode_trajectory_data, parameter) / time_delta; 
  p_no_spike_at_position = np.exp(-time_steps_per_bin * time_delta * spike_intensity)
  
  if verbose > 1:
    plt.figure(100); plt.clf();
    plt.plot(parameter['positions'], spike_intensity / spike_intensity.max())
    plt.plot(parameter['positions'], p_no_spike_at_position / p_no_spike_at_position.max())
  
  # timebins 
  decoding_spike_time_steps = decode_data['time_steps'];
  decoding_spikes_to_time_bin = np.asarray(decoding_spike_time_steps // time_steps_per_bin, dtype=int);
  time_bins_with_spikes, time_bins_start, spikes_to_time_bin_with_spikes, spike_counts = np.unique(decoding_spikes_to_time_bin, return_index=True, return_inverse=True, return_counts=True )
  time_bins_split = time_bins_start[1:];
  #n_time_bins = len(time_bins_with_spikes);
  
  # intensity ('kernel density estimate of the encoding data')
  intensities = intensity(encode_data, encode_trajectory_data, decode_data, parameter, normalize=True);
  if verbose:
    print(intensities.shape)
  if verbose > 1:
    plt.figure(101); plt.clf();
    plt.imshow(intensities.T, interpolation='none');
  
  # likelihood / observation distribution
  intensity_bins = np.split(intensities, time_bins_split);
  likelihoods = np.array([np.prod(i * time_delta, axis=0) * p_no_spike_at_position for i in intensity_bins]);
  if verbose > 1:
    plt.figure(102); plt.clf();
    plt.imshow(likelihoods.T, interpolation='none');
  
  #transition matrix
  transition_matrix = gaussian_transition_matrix(parameter);
  if verbose > 1:
    plt.figure(103); plt.clf();
    plt.imshow(transition_matrix, interpolation='none'); 
  
  
  # posteriors
  last_posterior = np.ones(len(transition_matrix))

  sample_rate = 1000/parameter['time_bin']
  n_time_steps = int(np.ceil(decode_total_time * sample_rate / time_steps_per_bin));
  time_bins_with_spikes = np.hstack([time_bins_with_spikes, [n_time_steps]]);
  posteriors = np.zeros((n_time_steps, likelihoods.shape[1]))
  
  i = 0;
  t_next = time_bins_with_spikes[i];
  for t in range(n_time_steps):
    if t < t_next:
      posteriors[t, :] = p_no_spike_at_position * np.matmul(transition_matrix, np.nan_to_num(last_posterior));
    else:
      posteriors[t, :] = likelihoods[i] * np.matmul(transition_matrix, np.nan_to_num(last_posterior));
      i += 1;
      t_next = time_bins_with_spikes[i];
    
    posteriors[t, :] = posteriors[t, :] / np.nansum(posteriors[t, :]);
                                                    
    if np.all(np.isnan(posteriors[t])):
      last_posterior = np.ones(len(transition_matrix))
    else:
      last_posterior = posteriors[t, :]
    
    if verbose:
      print('%d/%d' % (t, n_time_steps));
      
  if verbose > 1:
    plt.figure(104); plt.clf();
    plt.imshow(posteriors.T, interpolation='none')

  return likelihoods, posteriors
  
