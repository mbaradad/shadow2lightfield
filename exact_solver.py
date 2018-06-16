import numpy as np

class ExactSolver:
  def __init__(self, A_mat_fs, regularizer, Amat_fs_constructor, beta, ignore_region_indices):
    self.A_mat_fs = A_mat_fs
    self.Amat_fs_constructor = Amat_fs_constructor
    self.regularizer = regularizer
    self.beta = beta
    self.gain_mat = None
    self.ignore_region_indices = ignore_region_indices

  def construct_gain_mat(self):
    gain_mat = self.Amat_fs_constructor.construct_A_mat_gain_fs(self.beta, self.A_mat_fs,
                                                                         self.ignore_region_indices, self.regularizer)
    self.gain_mat = gain_mat
    return gain_mat

  def solve_inverse(self, obs):
    if self.gain_mat is None:
      self.gain_mat = self.construct_gain_mat()
    return np.matmul(self.gain_mat, obs)
