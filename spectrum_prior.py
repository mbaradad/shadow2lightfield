from LF_utils import get_LF_filter, get_frequencies, get_LF_spectrum_delta_space_frequencies
from utils import *
from scipy.sparse import diags
import math
import numpy as np
from tqdm import tqdm
import warnings

class SpectrumPriorGenerator():
  def __init__(self, LF_shape, LF_spectrum_shape, prior_args, params):
    self.LF_shape = LF_shape
    self.LF_spectrum_shape = LF_spectrum_shape
    self.prior_args = prior_args
    self.params = params

    LF_shape_with_channel = [i for i in LF_shape]
    LF_shape_with_channel.insert(0, 1)
    self.active_frequencies, self.fs_by_a = get_LF_filter(self.LF_spectrum_shape, LF_shape_with_channel, prior_args, params, return_fs_by_a=True)
    self.as_by_f = dict()
    for k in self.fs_by_a.keys():
      for i in self.fs_by_a[k]:
        self.as_by_f[i] = k

    if len(self.active_frequencies.shape) == 5:
      self.active_frequencies = self.active_frequencies[0, :, :, :, :]
    self.active_frequencies = np.asarray(self.active_frequencies, dtype='bool')

    #spatial spectrum deltas grouped taking into account the spatial padding
    all_f_spatial_indices = get_LF_spectrum_delta_space_frequencies(LF_spectrum_shape, zero_pad_in_space=prior_args['zero_pad_space'])

    active_fs = []
    print 'Computing prior frequencies'
    # if an element (n, m, l, k) is active in self.active_frequencies, we add the 4 deltas grouped if zero spatial padding is applied.
    for k in tqdm(all_f_spatial_indices):
      #get the angular set of frequencies active for all th e spatial deltas
      angular_fs = []
      for f in k:
        angular_fs.append(self.active_frequencies[f[0],f[1]])
      #logical or of all arrays
      active_angular_fs = np.array(angular_fs).sum(0) > 0
      index_active_angular_fs = np.where(active_angular_fs)
      index_active_angular_fs = zip(index_active_angular_fs[0], index_active_angular_fs[1])
      for r in index_active_angular_fs:
        active_fs.append([(f[0], f[1], r[0], r[1]) for f in k])

    self.active_frequencies_indices = active_fs
    self.solution_shape = len(self.active_frequencies_indices)
    self.unactive_components = None

    if prior_args['K_components'] > self.solution_shape:
      print 'Solving for ' + str(self.solution_shape) + ' (all) active frequencies though ' + str(prior_args['K_components']) + ' where specified'
    else:
      #only solve for the K components with highest variance
      n_all_active_frequencies = len(self.active_frequencies_indices)
      regularizer = self.get_regularizer(plot=False)
      freq_and_prior = [(regularizer.diagonal()[i], self.active_frequencies_indices[i]) for i in range(len(self.active_frequencies_indices))]

      self.solution_shape = prior_args['K_components']
      freq_and_prior.sort()
      self.active_frequencies_indices = [freq_and_prior[i][1] for i in range(self.solution_shape)]
      unactive_components = [freq_and_prior[i][1] for i in range(self.solution_shape, len(freq_and_prior))]

      for i in unactive_components:
        for k in i:
          self.active_frequencies[k] = 0
      self.solution_shape = prior_args['K_components']
      print 'Solving for ' + str(self.solution_shape) + ' active frequencies of a total of ' + str(n_all_active_frequencies) + ' in the 3D manifold.'

  def plot_simplified_prior(self, prior, name):
    nftheta = self.prior_args['nfreq'][2]
    nffi = self.prior_args['nfreq'][3]
    simplified_frequencies = np.zeros((self.LF_spectrum_shape[1], self.LF_spectrum_shape[2], nftheta, nffi),
                                      dtype='float32')
    all_frequencies = np.zeros(self.LF_spectrum_shape[1:], dtype='float32')
    flattened_i = 0
    for k in self.active_frequencies_indices:
      # take into account the zero pad in space
      for (fy, fx, ftheta, ffi) in k:
        all_frequencies[fy, fx, ftheta, ffi] = prior[flattened_i]
        if ftheta > nftheta or ffi > nffi:
          continue
        simplified_frequencies[fy, fx, ftheta, ffi] = prior[flattened_i]
      flattened_i = flattened_i + 1
    return

  def get_active_frequencies(self):
    return self.active_frequencies, self.active_frequencies_indices, self.solution_shape

  def get_regularizer(self, plot=False, postfix=''):
    #flattened_prior = self.get_1_over_spatial_f_regularizer()
    if not self.as_by_f is None:
      flattened_prior = self.get_regularizer_with_propagation_scaling()
    else:
      flattened_prior = self.get_1_over_spatial_f_regularizer()
    return flattened_prior

  def get_power_regularizer(self):
    return np.eye(self.solution_shape[0])

  def get_1_over_spatial_f_regularizer(self, gamma=2):
    fy, fx, fv, fu = get_frequencies(self.LF_spectrum_shape[1:])
    fs = np.meshgrid(fy, fx, fv, fu, indexing='ij', copy=False)

    flattened_i = 0
    f = np.zeros(self.solution_shape)
    for k in self.active_frequencies_indices:
      # just use first index of frequency, a k is the list with the multiple frequencies that compose a single impulse response,
      # which be more than one if using zeropadding in the primal domain
      (fy, fx, ftheta, ffi) = k[0]
      # f[flattened_i] = np.sqrt(fs[0][fy,fx,ftheta,ffi] ** 2 + fs[1][fy,fx,ftheta,ffi] ** 2 +
      #                          6000 * fs[2][fy,fx,ftheta,ffi] ** 2 + 6000 * fs[3][fy,fx,ftheta,ffi] ** 2)
      f[flattened_i] = np.sqrt(fs[0][fy, fx, ftheta, ffi] ** 2 + fs[1][fy, fx, ftheta, ffi] ** 2)

      flattened_i = flattened_i + 1

    cv = 1 / f ** gamma
    cv[cv == np.inf] = 0

    flattened_reg_factors = 1 / cv
    flattened_reg_factors[flattened_reg_factors == np.inf] = 0

    # normalize regularization, so beta hasn't to be finetuned for each
    flattened_reg_factors = flattened_reg_factors / np.sqrt((flattened_reg_factors ** 2).sum())

    return diags(flattened_reg_factors)

  # full prior as specified in the paper
  def get_regularizer_with_propagation_scaling(self, gamma=2):
    fys, fxs, fvs, fus = get_frequencies(self.LF_spectrum_shape[1:], self.params)

    cvs = np.zeros(self.solution_shape)
    flattened_i = 0

    a_s = list()
    for k in self.active_frequencies_indices:
      # just use first index of frequency
      (fy, fx, ftheta, ffi) = k[0]
      fy = fys[fy]
      fx = fxs[fx]
      a = None
      for i in k:
          try:
            a = self.as_by_f[i]
            break
          except:
            continue
      if a is None:
        print "a not found for : " + str(k)
        a = 0
      f_space = np.sqrt(fy ** 2 + fx ** 2)
      f_space_a = (f_space * (1 / (1 + a)))
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LF_a = (1 / (1 + a)) * 1 / f_space_a ** gamma
      if math.isinf(LF_a) or math.isnan(LF_a):
        LF_a = 0
      cvs[flattened_i] = LF_a
      flattened_i = flattened_i + 1
      # normalize regularization, so beta hasn't to be finetuned for each
    cvs = cvs / np.sqrt((cvs ** 2).sum())
    cvs[cvs == np.inf] = 0

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      flattened_reg_factors = 1 / cvs
    flattened_reg_factors[flattened_reg_factors == np.inf] = 0

    # normalize regularization, so beta hasn't to be finetuned for each
    flattened_reg_factors = np.abs(flattened_reg_factors / np.sqrt((flattened_reg_factors ** 2).sum()))

    return diags(flattened_reg_factors)