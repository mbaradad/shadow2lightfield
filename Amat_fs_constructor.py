from tensorflow_utils import Tensorflow_Utils
from utils import *
import time
from tqdm import tqdm

#constructs the transfer in the fourier spectrum
class Amat_fs_constructor:
  def __init__(self, A_mat, A_mat_filename, LF_shape, LF_spectrum_shape, obs_shape, active_frequencies_indices, args, solution_shape, cache_dir):
    self.A_mat = A_mat
    self.A_mat_filename = A_mat_filename
    self.cache_dir = cache_dir
    self.tf_utils = None
    self.LF_shape = LF_shape
    self.LF_spectrum_shape = LF_spectrum_shape
    self.obs_shape = obs_shape
    self.active_frequencies_indices = active_frequencies_indices
    self.solution_shape = solution_shape
    self.args = args

    #stores precomputed bases and transfer functions
    self.A_mat_fs_bases_dir = None
    self.A_mat_fs_inverse_bases_dir = None

    self.A_mat_fs = None
    self.A_mat_fs_inverse = None
    self.A_mat_gain_fs = None

  def _get_tensorflow_utils(self):
    if self.tf_utils is None:
      self.tf_utils = Tensorflow_Utils(self.A_mat[0], self.LF_shape)
    return self.tf_utils

  def get_A_mat_fs_bases_hash_params(self):
    additional_args = dict()
    #we recompute amat_fs if the propagation is computed
    if self.args['compute_propagation']:
      additional_args['compute_propagation'] = True
    return additional_args

  def get_A_mat_fs_inverse_bases_hash_params(self, LF_subsample):
    #the inverse bases depends on the subsample factor
    additional_args = dict()
    additional_args['LF_subsample'] = LF_subsample
    return additional_args

  def get_cache_dir(self, prefix, params):
    mkdir(self.cache_dir)
    cache_dir = self.cache_dir + '/' + self.A_mat_filename.split('.')[0]
    mkdir(cache_dir)
    postfix = 'shape_' + ''.join([str(k) + '_' for k in self.LF_shape]) + ''.join([k + '_' + str(params[k]) for k in params.keys()])
    postfix = postfix[:-1]
    cache_dir = cache_dir + '/' + prefix + '_' + postfix
    mkdir(cache_dir)
    return cache_dir

  def get_A_mat_fs_inverse_bases_dir(self, LF_subsample_angular):
    if self.A_mat_fs_inverse_bases_dir is None:
      self.A_mat_fs_inverse_bases_dir = self.get_cache_dir('A_mat_fs_inverse_bases', self.get_A_mat_fs_inverse_bases_hash_params(LF_subsample_angular))
    return self.A_mat_fs_inverse_bases_dir

  def get_A_mat_fs_bases_dir(self):
    if self.A_mat_fs_bases_dir is None:
      self.A_mat_fs_bases_dir = self.get_cache_dir('A_mat_fs_bases', self.get_A_mat_fs_bases_hash_params())
    return self.A_mat_fs_bases_dir

  def get_A_mat_fs_inverse_base(self, LF_pos, LF_subsample_angular, LF_subsample_indices):
    [LF_subsample_indices_u, LF_subsample_indices_v] = LF_subsample_indices
    bases_dir = self.get_A_mat_fs_inverse_bases_dir(LF_subsample_angular)
    actual_base_str = str(LF_pos)
    actual_base_filename = bases_dir + '/' + actual_base_str
    if os.path.exists(actual_base_filename):
      try:
        actual_base = load_np_array(actual_base_filename)
        self.A_mat_inverse_precomptued_bases = self.A_mat_inverse_precomptued_bases + 1
        return actual_base
      except: pass
    # compute, store and return
    # using gpu is considerably faster than cpu (10xspeedup aprox)
    actual_row = self._get_tensorflow_utils().get_obs_from_LF_delta(self.LF_shape, LF_pos,
                                                              LF_spectrum_shape=self.LF_spectrum_shape, return_base=True)
    actual_row = actual_row.reshape(self.LF_shape)[:,:,LF_subsample_indices_u,:][:,:,:, LF_subsample_indices_v]
    actual_base = actual_row.flatten()
    save_np_array(actual_base, actual_base_filename)
    return actual_base

  def get_A_mat_fs_base(self, LF_pos):
    bases_dir = self.get_A_mat_fs_bases_dir()
    actual_base_str = '_'.join([str(i) for i in LF_pos])
    actual_base_filename = bases_dir + '/' + actual_base_str
    if os.path.exists(actual_base_filename):
      try:
        actual_base = load_np_array(actual_base_filename)
        self.A_mat_precomptued_bases = self.A_mat_precomptued_bases + 1
        return actual_base
      except: pass
    actual_row, LF_base = self._get_tensorflow_utils().get_obs_from_LF_delta(self.LF_shape, LF_pos,
                                                         LF_spectrum_shape=self.LF_spectrum_shape)
    actual_base = actual_row.flatten()
    save_np_array(actual_base, actual_base_filename)
    return actual_base

  def construct_A_mat_fs_inverse(self, LF_subsample_angular=30):
    LF_subsample_indices_u = range(self.LF_shape[2])[::LF_subsample_angular]
    LF_subsample_indices_v = range(self.LF_shape[3])[::LF_subsample_angular]
    self.LF_inverse_shape = (self.LF_shape[0], self.LF_shape[1], len(LF_subsample_indices_u),
                             len(LF_subsample_indices_v))
    if not self.A_mat_fs_inverse is None:
      return self.A_mat_fs_inverse, self.LF_inverse_shape
    print 'Computing inverse mat'

    self.A_mat_fs_inverse = np.zeros((reduce(lambda x, y: x * y, self.LF_inverse_shape), self.solution_shape), dtype='complex64')
    self.A_mat_inverse_precomptued_bases = 0
    flattened_i = 0
    starting_time = time.time()
    for LF_pos in tqdm(self.active_frequencies_indices):
      self.A_mat_fs_inverse[:, flattened_i] = self.get_A_mat_fs_inverse_base(LF_pos, LF_subsample_angular, [LF_subsample_indices_u, LF_subsample_indices_v])
      flattened_i = flattened_i + 1
    print 'End computing inverse amat fs ' + str(flattened_i) + ' of ' + str(self.A_mat_fs_inverse.shape[1])
    return self.A_mat_fs_inverse, self.LF_inverse_shape

  def construct_A_mat_fs(self):
    self.A_mat_precomptued_bases = 0
    if self.A_mat_fs is None:
      self.A_mat_fs = np.zeros((self.A_mat[0].shape[0], self.solution_shape),dtype='complex64')
      flattened_i = 0
      print 'Computing Amat fs...'
      # TODO: This could be easily parallelized
      for LF_pos in tqdm(self.active_frequencies_indices):
        actual_row = self.get_A_mat_fs_base(LF_pos)
        self.A_mat_fs[:, flattened_i] = actual_row
        flattened_i = flattened_i + 1
    return self.A_mat_fs

  def construct_A_mat_gain_fs(self, beta, A_mat_fs, ignore_region_indices, regularizer):
    if self.A_mat_gain_fs is None:
      print 'Computing gain mat...'
      print 'This can take a while...'
      print '~180s using 10K components and 150x225 observations samples, with an i6700K cpu '
      starting_time = time.time()
      if not ignore_region_indices is None:
        A_mat_fs[ignore_region_indices, :] = 0
      reg_mat = regularizer
      A_H = np.conjugate(np.transpose(A_mat_fs))
      A_H_A = np.matmul(A_H, A_mat_fs)
      W_H = np.asarray(np.conjugate(np.transpose(reg_mat)).todense(), dtype='float32')
      W_H_W = np.asarray(W_H * reg_mat.todense(), dtype='float32')
      inv_A_H_A_plus_beta_W_H_W = np.linalg.inv(A_H_A + beta*W_H_W)
      #condition_number = np.linalg.cond(inv_A_H_A_plus_beta_W_H_W)
      self.A_mat_gain_fs = np.asarray(np.matmul(inv_A_H_A_plus_beta_W_H_W, A_H))
      time_spent = time.time() - starting_time
      print 'Time to compute gain mat: ' + str(time_spent)
    return self.A_mat_gain_fs