from Amat_fs_constructor import Amat_fs_constructor
from LF_utils import pad_LF
from exact_solver import ExactSolver
from spectrum_prior import SpectrumPriorGenerator
import numpy as np
import time

class Solver(object):
  def __init__(self, A_mat, args, LF_shape, ignore_region_mask, params, ifft_inverse=False):
    self.args = args
    prior_args = dict()
    prior_args['a0'] = [float(args['a0'])]
    prior_args['a1'] = [float(args['a1'])]
    prior_args['K_components'] = args['K_components']
    prior_args['angular_mirror_padding'] = [0.5,0.5]
    prior_args['zero_pad_space'] = True
    prior_args["filter"] = ["slope"]

    self.prior_args = prior_args

    LF_shape_with_channel = [i for i in LF_shape]
    LF_shape_with_channel.insert(0,1)
    LF_spectrum_shape = pad_LF(np.zeros(LF_shape_with_channel), prior_args, return_shape_only=True)

    self.params = params
    self.spectrum_prior_generator = SpectrumPriorGenerator(LF_shape, LF_spectrum_shape, prior_args, params)
    self.active_frequencies, self.active_frequencies_indices, self.solution_shape = self.spectrum_prior_generator.get_active_frequencies()
    self.spectrum_prior = self.spectrum_prior_generator.get_regularizer()

    self.LF_shape = LF_shape
    self.obs_shape = (LF_shape[2], LF_shape[3])
    self.LF_spectrum_shape = LF_spectrum_shape
    self.A_mat = A_mat
    self.ignore_region_mask = ignore_region_mask
    if not ignore_region_mask is None:
      self.ignore_region_indices = np.where(ignore_region_mask != 1)
    else:
      self.ignore_region_indices = None
    self.A_mat_fs = None
    self.A_mat_fs_inverse = None
    self.last_beta = None
    A_mat_filename = args['amat'].split('/')[-1]
    self.Amat_fs_constructor = Amat_fs_constructor(self.A_mat, A_mat_filename, LF_shape, LF_spectrum_shape, self.obs_shape,
                                                   self.active_frequencies_indices, self.args, self.solution_shape, args['cache_dir'])

    self.inverse_LF_angular_subsample = args['result_subsample']
    self.ifft_inverse = ifft_inverse

  def construct_A_mat_fs(self):
    if self.A_mat_fs is None:
      self.A_mat_fs = self.Amat_fs_constructor.construct_A_mat_fs()
    return self.A_mat_fs

  def construct_A_mat_fs_inverse(self):
    if self.A_mat_fs_inverse is None:
      self.A_mat_fs_inverse, self.LF_inverse_shape = self.Amat_fs_constructor.construct_A_mat_fs_inverse(self.inverse_LF_angular_subsample)
    return self.A_mat_fs_inverse, self.LF_inverse_shape

  def solve_inverse(self, obs, beta):
    flattened_obs = obs.reshape([-1, 1])
    Amat_fs = self.construct_A_mat_fs()
    if not self.ifft_inverse:
      inverse_Amat_fs, inverse_LF_shape = self.construct_A_mat_fs_inverse()

    if self.last_beta is None or self.last_beta != beta:
      #reuse solver if possible to a`void doing the inversion each time
      self.solver = ExactSolver(Amat_fs, self.spectrum_prior, self.Amat_fs_constructor, beta, self.ignore_region_indices)
      self.last_beta = beta

    spectrum_solution = self.solver.solve_inverse(flattened_obs)

    if self.ifft_inverse:
      print 'Computing LF with ifft. This can take a while'
      print '~240s using 10K components and 25x35x150x225 Light field'
      starting_time = time.time()
      full_LF = self.get_full_LF(spectrum_solution)
      time_spent = time.time() - starting_time
      print 'Time to compute spectrum LF ifft: ' + str(time_spent)
      LF = full_LF[:,:,:,::self.inverse_LF_angular_subsample,::self.inverse_LF_angular_subsample]
    else:
      LF = np.expand_dims(np.reshape(np.matmul(inverse_Amat_fs, spectrum_solution), inverse_LF_shape), 0)
    LF = LF.real
    return LF, spectrum_solution


  def get_full_LF(self, spectrum_solution):
    complete_spectrum = np.zeros(self.LF_spectrum_shape, dtype='complex128')
    flattened_i = 0
    for k in self.active_frequencies_indices:
      #take into account the zero pad in space
      for (fy, fx, ftheta, ffi) in k:
        complete_spectrum[:, fy, fx, ftheta, ffi] = complete_spectrum[:, fy, fx, ftheta, ffi] + spectrum_solution[flattened_i]
      flattened_i = flattened_i + 1

    print 'Computing complete spectrum with ifft:...'
    print '~180s using 10K components and 150x225 observations samples, with an i6700K cpu '
    ifft = np.fft.ifftn(complete_spectrum)
    print 'Time to compute ifft with shape ' + str(complete_spectrum.shape) + ': ' + str()
    LF_padded = np.real(ifft)

    offsets = [complete_spectrum.shape[i+1] / 2 - self.LF_shape[i] / 2 for i in range(4)]

    LF = np.zeros(self.LF_shape)
    if offsets[0] != 0:
      #undo zeropadding
      odd_0 = self.LF_shape[0] % 2
      odd_1 = self.LF_shape[1] % 2

      LF[:offsets[0],:offsets[1],:,:] = LF_padded[:,-offsets[0]:, -offsets[1]:,
           offsets[2]:offsets[2] + self.LF_shape[2],
           offsets[3]:offsets[3] + self.LF_shape[3]]
      LF[-(offsets[0] + odd_0):,:offsets[1],:,:] = LF_padded[:,:(offsets[0] + odd_0), -offsets[1]:,
           offsets[2]:offsets[2] + self.LF_shape[2],
           offsets[3]:offsets[3] + self.LF_shape[3]]
      LF[:offsets[0],-(offsets[1] + odd_1):,:,:] = LF_padded[:,-offsets[0]:, :(offsets[1] + odd_1),
           offsets[2]:offsets[2] + self.LF_shape[2],
           offsets[3]:offsets[3] + self.LF_shape[3]]
      LF[-(offsets[0] + odd_0):,-(offsets[1] + odd_1):,:,:] = LF_padded[:,:(offsets[0] + odd_0), :(offsets[1] + odd_1),
           offsets[2]:offsets[2] + self.LF_shape[2],
           offsets[3]:offsets[3] + self.LF_shape[3]]
    else:
      LF[:, :, :, :] = LF_padded[:, :, :,
                                             offsets[2]:offsets[2] + self.LF_shape[2],
                                             offsets[3]:offsets[3] + self.LF_shape[3]]
    LF = np.expand_dims(LF, 0)
    return LF